//! Transactions whose `begin` is cancelled or fails after its `BEGIN` or
//! `SAVEPOINT` was sent. See <https://github.com/transact-rs/sqlx/issues/4393>.
//!
//! Each test parks the connection behind a statement blocked on an advisory
//! lock held by an observer connection (a [`Gate`]), so the server leaves the
//! `BEGIN` unanswered for exactly as long as the test needs. Durability is
//! checked from the observer, never from the connection under test.

use std::env;
use std::time::Duration;

use futures::future::{select, Either};
use futures::poll;
use sqlx::postgres::{PgConnection, PgPool, PgPoolOptions, PgTransactionManager, Postgres};
use sqlx::{Connection, Executor, Transaction, TransactionManager};
use sqlx_test::{new, setup_if_needed};

#[sqlx_macros::test]
async fn it_rolls_back_a_begin_cancelled_in_flight() -> anyhow::Result<()> {
    let mut observer = new::<Postgres>().await?;
    let table = create_table(&mut observer, "cancelled_begin").await?;
    let pool = single_connection_pool().await?;

    let mut conn = pool.acquire().await?;
    let pid = backend_pid(&mut conn).await?;
    let gate = Gate::close(&mut observer, &mut conn).await?;

    // What `Pool::begin` does after `acquire`.
    let mut begin = Transaction::<Postgres>::begin(conn, None);
    assert!(poll!(&mut begin).is_pending());
    drop(begin);

    gate.open(&mut observer).await?;
    assert_pool_commits(pool, &mut observer, pid, &table).await
}

/// SeaORM drives `TransactionManager` directly rather than through `Transaction`.
#[sqlx_macros::test]
async fn it_rolls_back_a_transaction_manager_begin_cancelled_in_flight() -> anyhow::Result<()> {
    let mut observer = new::<Postgres>().await?;
    let table = create_table(&mut observer, "cancelled_manager_begin").await?;
    let pool = single_connection_pool().await?;

    let mut conn = pool.acquire().await?;
    let pid = backend_pid(&mut conn).await?;
    let gate = Gate::close(&mut observer, &mut conn).await?;

    let mut begin = PgTransactionManager::begin(&mut conn, None);
    assert!(poll!(&mut begin).is_pending());
    drop(begin);
    drop(conn);

    gate.open(&mut observer).await?;
    assert_pool_commits(pool, &mut observer, pid, &table).await
}

/// No cancellation: `begin` returns the error of a statement pipelined ahead
/// of its `BEGIN`, which the server still executes afterward.
#[sqlx_macros::test]
async fn it_rolls_back_a_begin_failed_by_an_earlier_statement() -> anyhow::Result<()> {
    let mut observer = new::<Postgres>().await?;
    let table = create_table(&mut observer, "begin_failed_early").await?;
    let pool = single_connection_pool().await?;

    let mut conn = pool.acquire().await?;
    let pid = backend_pid(&mut conn).await?;
    let gate = Gate::close(&mut observer, &mut conn).await?;

    let mut begin = Transaction::<Postgres>::begin(conn, None);
    assert!(poll!(&mut begin).is_pending());

    // Fail the blocked statement, as a `statement_timeout` would.
    let cancelled: bool = sqlx::query_scalar("SELECT pg_cancel_backend($1)")
        .bind(pid)
        .fetch_one(&mut observer)
        .await?;
    assert!(cancelled);
    let err = begin
        .await
        .expect_err("BEGIN should fail with the cancelled statement");
    assert_eq!(
        sqlstate(&err).as_deref(),
        Some("57014"),
        "unexpected error: {err}"
    );

    gate.open(&mut observer).await?;
    assert_pool_commits(pool, &mut observer, pid, &table).await
}

#[sqlx_macros::test]
async fn it_recovers_from_a_rejected_begin() -> anyhow::Result<()> {
    let mut observer = new::<Postgres>().await?;
    let table = create_table(&mut observer, "rejected_begin").await?;
    let pool = single_connection_pool().await?;
    let pid = backend_pid(&mut *pool.acquire().await?).await?;

    let err = pool
        .begin_with("BEGIN ISOLATION LEVEL NONSENSE")
        .await
        .expect_err("BEGIN should be rejected");
    assert_eq!(
        sqlstate(&err).as_deref(),
        Some("42601"),
        "unexpected error: {err}"
    );

    assert_pool_commits(pool, &mut observer, pid, &table).await
}

#[sqlx_macros::test]
async fn it_rolls_back_a_begin_that_fails_after_opening_a_block() -> anyhow::Result<()> {
    let mut observer = new::<Postgres>().await?;
    let table = create_table(&mut observer, "begin_failed_late").await?;
    let pool = single_connection_pool().await?;
    let pid = backend_pid(&mut *pool.acquire().await?).await?;

    let err = pool
        .begin_with("BEGIN; SELECT 1/0")
        .await
        .expect_err("BEGIN should fail");
    assert_eq!(
        sqlstate(&err).as_deref(),
        Some("22012"),
        "unexpected error: {err}"
    );

    assert_pool_commits(pool, &mut observer, pid, &table).await
}

/// Control: a cancelled statement inside an open transaction.
#[sqlx_macros::test]
async fn it_rolls_back_a_dropped_transaction_with_a_cancelled_statement() -> anyhow::Result<()> {
    let mut observer = new::<Postgres>().await?;
    let table = create_table(&mut observer, "cancelled_statement").await?;
    let pool = single_connection_pool().await?;

    let mut tx = pool.begin().await?;
    let pid = backend_pid(&mut tx).await?;
    insert(&mut tx, &table, 1).await?;
    let gate = Gate::close(&mut observer, &mut tx).await?;
    drop(tx);

    gate.open(&mut observer).await?;
    assert_pool_commits(pool, &mut observer, pid, &table).await
}

#[sqlx_macros::test]
async fn it_keeps_the_outer_transaction_when_a_nested_begin_is_cancelled_then_commit(
) -> anyhow::Result<()> {
    cancel_nested_begin("cancelled_nested_commit", true).await
}

#[sqlx_macros::test]
async fn it_keeps_the_outer_transaction_when_a_nested_begin_is_cancelled_then_rollback(
) -> anyhow::Result<()> {
    cancel_nested_begin("cancelled_nested_rollback", false).await
}

async fn cancel_nested_begin(name: &str, commit: bool) -> anyhow::Result<()> {
    let mut observer = new::<Postgres>().await?;
    let table = create_table(&mut observer, name).await?;
    let mut conn = new::<Postgres>().await?;

    let mut tx = conn.begin().await?;
    insert(&mut tx, &table, 1).await?;
    let gate = Gate::close(&mut observer, &mut tx).await?;

    let mut nested = Connection::begin(&mut *tx);
    assert!(poll!(&mut nested).is_pending());
    drop(nested);

    gate.open(&mut observer).await?;
    insert(&mut tx, &table, 2).await?;
    assert_eq!(
        rows(&mut observer, &table).await?,
        Vec::<i32>::new(),
        "outer transaction's writes are visible before it ended"
    );

    let expected: &[i32] = if commit {
        tx.commit().await?;
        &[1, 2]
    } else {
        tx.rollback().await?;
        &[]
    };
    assert_eq!(rows(&mut observer, &table).await?, expected);
    Ok(())
}

#[sqlx_macros::test]
async fn it_keeps_an_aborted_outer_transaction_when_a_nested_begin_fails() -> anyhow::Result<()> {
    let mut observer = new::<Postgres>().await?;
    let table = create_table(&mut observer, "rejected_nested").await?;
    let mut conn = new::<Postgres>().await?;

    let mut tx = conn.begin().await?;
    insert(&mut tx, &table, 1).await?;
    tx.execute("SELECT 1/0").await.expect_err("should abort");
    let err = Connection::begin(&mut *tx)
        .await
        .expect_err("SAVEPOINT should be rejected");
    assert_eq!(
        sqlstate(&err).as_deref(),
        Some("25P02"),
        "unexpected error: {err}"
    );

    // The outer transaction is still aborted, not silently ended.
    let write = insert(&mut tx, &table, 2).await;
    assert!(write.is_err(), "write ran outside the outer transaction");
    tx.rollback().await?;

    assert_eq!(rows(&mut observer, &table).await?, Vec::<i32>::new());
    Ok(())
}

/// An advisory lock held by the observer, on which a statement on the
/// connection under test is left blocked. Anything queued after it on that
/// connection is sent, but unanswered until the gate opens.
struct Gate {
    key: i64,
}

impl Gate {
    async fn close(observer: &mut PgConnection, conn: &mut PgConnection) -> anyhow::Result<Self> {
        let pid = backend_pid(conn).await?;
        let key = i64::from(pid);
        sqlx::query("SELECT pg_advisory_lock($1)")
            .bind(key)
            .execute(&mut *observer)
            .await?;

        let sql = format!("SELECT pg_advisory_xact_lock({key})");
        let blocked = conn.execute(&*sql);
        let waiting = Box::pin(wait_for_advisory_lock(observer, pid));
        match select(blocked, waiting).await {
            Either::Left((result, _)) => anyhow::bail!("statement was not blocked: {result:?}"),
            // Dropping the blocked statement leaves its response in flight.
            Either::Right((result, _blocked)) => result?,
        }
        Ok(Self { key })
    }

    async fn open(self, observer: &mut PgConnection) -> anyhow::Result<()> {
        let unlocked: bool = sqlx::query_scalar("SELECT pg_advisory_unlock($1)")
            .bind(self.key)
            .fetch_one(observer)
            .await?;
        anyhow::ensure!(unlocked, "gate was not closed");
        Ok(())
    }
}

/// Asserts that an autocommit write through the pool is durable, and that the
/// connection with backend `pid` was returned to the pool outside any
/// transaction (or discarded).
async fn assert_pool_commits(
    pool: PgPool,
    observer: &mut PgConnection,
    pid: i32,
    table: &str,
) -> anyhow::Result<()> {
    let mut conn = pool.acquire().await?;
    let reused = backend_pid(&mut conn).await? == pid;
    insert(&mut conn, table, 100).await?;
    drop(conn);
    assert_eq!(
        rows(observer, table).await?,
        [100],
        "pooled write is not visible"
    );

    if reused {
        let state = backend_state(observer, pid).await?;
        assert_eq!(
            state.as_deref(),
            Some("idle"),
            "pooled connection is in a transaction"
        );
    } else {
        // Discarding the connection is also safe, once its session has ended.
        sqlx_core::rt::timeout(Duration::from_secs(10), async {
            while backend_state(observer, pid).await?.is_some() {
                sqlx_core::rt::sleep(Duration::from_millis(5)).await;
            }
            anyhow::Ok(())
        })
        .await
        .map_err(|_| anyhow::anyhow!("discarded session {pid} never ended"))??;
    }

    pool.close().await;
    assert_eq!(rows(observer, table).await?, [100], "pooled write was lost");
    Ok(())
}

async fn single_connection_pool() -> anyhow::Result<PgPool> {
    setup_if_needed();
    let pool = PgPoolOptions::new()
        .max_connections(1)
        .min_connections(0)
        .connect(&env::var("DATABASE_URL")?)
        .await?;
    Ok(pool)
}

async fn create_table(observer: &mut PgConnection, name: &str) -> anyhow::Result<String> {
    let table = format!("sqlx_tx_{name}");
    observer
        .execute(&*format!(
            "DROP TABLE IF EXISTS {table}; CREATE TABLE {table} (id INTEGER NOT NULL)"
        ))
        .await?;
    Ok(table)
}

async fn insert(conn: &mut PgConnection, table: &str, id: i32) -> sqlx::Result<()> {
    sqlx::query(&format!("INSERT INTO {table} VALUES ($1)"))
        .bind(id)
        .execute(conn)
        .await?;
    Ok(())
}

async fn rows(observer: &mut PgConnection, table: &str) -> sqlx::Result<Vec<i32>> {
    sqlx::query_scalar(&format!("SELECT id FROM {table} ORDER BY id"))
        .fetch_all(observer)
        .await
}

async fn backend_pid(conn: &mut PgConnection) -> sqlx::Result<i32> {
    sqlx::query_scalar("SELECT pg_backend_pid()")
        .fetch_one(conn)
        .await
}

async fn backend_state(observer: &mut PgConnection, pid: i32) -> sqlx::Result<Option<String>> {
    sqlx::query_scalar("SELECT state FROM pg_stat_activity WHERE pid = $1")
        .bind(pid)
        .fetch_optional(observer)
        .await
}

async fn wait_for_advisory_lock(observer: &mut PgConnection, pid: i32) -> anyhow::Result<()> {
    sqlx_core::rt::timeout(Duration::from_secs(10), async {
        loop {
            let waiting: bool = sqlx::query_scalar(
                "SELECT EXISTS (SELECT 1 FROM pg_stat_activity \
                 WHERE pid = $1 AND wait_event_type = 'Lock' AND wait_event = 'advisory')",
            )
            .bind(pid)
            .fetch_one(&mut *observer)
            .await?;
            if waiting {
                return anyhow::Ok(());
            }
            sqlx_core::rt::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .map_err(|_| anyhow::anyhow!("session {pid} never blocked on the advisory lock"))?
}

fn sqlstate(err: &sqlx::Error) -> Option<String> {
    Some(err.as_database_error()?.code()?.into_owned())
}

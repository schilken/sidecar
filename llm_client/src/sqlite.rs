use std::fs;
use std::path::Path;

use sqlx::SqlitePool;

use crate::{clients::types::LLMClientError, config::LLMBrokerConfiguration};

pub async fn init(config: LLMBrokerConfiguration) -> Result<SqlitePool, LLMClientError> {
    let data_dir = config.data_dir.to_string_lossy().to_owned();

    if let Err(e) = fs::create_dir_all(data_dir.as_ref()) {
        println!("Failed to create data directory: {e}");
        return Err(LLMClientError::SqliteSetupError);
    }

    match connect(&data_dir).await {
        Ok(pool) => Ok(pool),
        Err(_) => {
            reset(&data_dir)?;
            connect(&data_dir).await
        }
    }
}

async fn connect(data_dir: &str) -> Result<SqlitePool, LLMClientError> {
    let url = format!("sqlite://{data_dir}/llm_data.data?mode=rwc");

    let pool = match SqlitePool::connect(&url).await {
        Ok(p) => {
            println!("Successfully established SQLite connection pool");
            p
        }
        Err(e) => {
            println!("Failed to create SQLite connection pool: {e}");
            return Err(LLMClientError::TokioMpscSendError);
        }
    };

    println!("Running database migrations...");
    if let Err(e) = sqlx::migrate!().run(&pool).await {
        println!("Migration failed: {e}");
        println!("Closing pool due to migration failure");
        pool.close().await;
        println!("Pool closed successfully");
        Err(LLMClientError::SqliteSetupError)
    } else {
        println!("Database migrations completed successfully");
        Ok(pool)
    }
}

fn reset(data_dir: &str) -> Result<(), LLMClientError> {
    let db_path = Path::new(data_dir).join("llm_data.data");
    let bk_path = db_path.with_extension("llm_data.bk");
    std::fs::rename(db_path, bk_path).map_err(|_| LLMClientError::SqliteSetupError)
}

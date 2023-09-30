// This is where we will create the default webserver for running the binary
// locally

use anyhow::Result;
use axum::routing::get;
use axum::Extension;
use clap::Parser;
use sidecar::{
    application::{application::Application, config::configuration::Configuration},
    webserver::repos,
};
use std::net::SocketAddr;
use tower_http::{catch_panic::CatchPanicLayer, cors::CorsLayer};

pub type Router<S = Application> = axum::Router<S>;

#[tokio::main]
async fn main() -> Result<()> {
    let configuration = Configuration::parse();
    // We get the logging setup first
    Application::install_logging(&configuration);

    // We initialize the logging here
    let application = Application::initialize(configuration).await?;
    let _ = start(application).await;
    println!("Hello world! application");
    Ok(())
}

// TODO(skcd): Create a new endpoint here which can start the sync for the
// whole repo and make that work
// you will also figure out how to keep the state and everything here by doing
// that.
// once that's done, we also need to change the indexing logic so we use the
// most frequently edited files first and then the rest (very important) and use
// that for scoring
pub async fn start(app: Application) -> anyhow::Result<()> {
    let bind = SocketAddr::new(app.config.host.parse()?, app.config.port);
    let api = Router::new()
        .route("/config", get(sidecar::webserver::config::get))
        .nest("/repo", repo_router());

    let api = api
        .layer(Extension(app.clone()))
        .with_state(app.clone())
        .with_state(app.clone())
        .layer(CorsLayer::permissive())
        .layer(CatchPanicLayer::new());

    let router = Router::new().nest("/api", api);

    axum::Server::bind(&bind)
        .serve(router.into_make_service())
        .await?;

    Ok(())
}

fn repo_router() -> Router {
    use axum::routing::*;
    Router::new().route("/sync", get(sidecar::webserver::repos::sync))
}
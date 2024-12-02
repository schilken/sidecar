# We are going to use this script to package up the binary as there are
# certain things we have to copy at the right place for this binary to work.
# - we need to have the ort runtime checked in, along with the model which is
# required.
# - and we also pass the dydlib library as well

# We can use this command to zip the whole thing together for the platform we are
# interetsed in, right now we will lock it to just mac
cargo build --bin webserver --release
zip -r sidecar target/release/webserver
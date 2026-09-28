# Local deployment scope

RoboCurate reads files chosen by the local operator and serves a browser UI on `127.0.0.1`. It does not implement multi-user authentication, authorization or internet-facing deployment. Do not expose the service port to an untrusted network.

Use only model checkpoints from a trusted source. Checkpoint loading and scripts are executable software dependencies. Do not include access tokens or raw recordings in issue reports.

No public security contact is configured while repository ownership and source licensing are pending. A maintainer should configure a private reporting channel before public release.

## Device management (collection-hub)

The optional collection-hub opens a node endpoint (default port 8422) to the collection LAN. Heartbeats require a per-device bearer token (stored as a SHA-256 hash); `/install/*` serves the node script without authentication. Its admin API binds to `127.0.0.1` only. RoboCurate forwards a fixed set of admin actions under `/api/hub/*` with its usual same-origin checks, so anyone who can reach a LAN-bound RoboCurate can also pause or trigger sync, register devices and, when a node was installed with `--allow-cleanup`, confirm deletion of already verified data on that device. Traffic is plain HTTP; run it only on a network you trust.

The hub pulls data with a dedicated SSH key that each device restricts to `rrsync -ro` on its recordings directory. Node-reported paths are validated before they are used on the server, and rsync does not copy symlinks or device files. Remote shutdown and device-side cleanup are disabled unless the node is started with `--allow-shutdown` / `--allow-cleanup`.

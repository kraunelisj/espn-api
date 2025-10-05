# Deploying the ESPN Fantasy MCP backend to AWS App Runner

This guide packages the `espn_api` MCP server together with the [OpenWebUI MCPO
proxy](https://docs.openwebui.com/openapi-servers/mcp) and deploys it on
[AWS App Runner](https://aws.amazon.com/apprunner/).  App Runner keeps the
container running, provides an HTTPS endpoint, and automatically redeploys the
service when you push a new image tag.

> **Tip:** The instructions below assume you already deployed the frontend with
> AWS Amplify.  Once App Runner exposes the backend URL you can point the
> frontend to it by setting the `VITE_BASE_URL` environment variable (see
> [section 5](#5-share-the-backend-url-with-the-frontend)).

## 1. Prerequisites

* An AWS account with permissions for Elastic Container Registry (ECR) and App
  Runner.
* AWS CLI v2 configured locally (`aws configure`).
* Docker installed locally so you can build and push the container image.
* Access to an ESPN league plus the `SWID`/`espn_s2` cookies that the MCP tools
  require.
* (Optional) A custom domain managed in Route 53 if you plan to add one later.

## 2. Package the backend

The repository now contains a production `Dockerfile` and entrypoint that start
OpenWebUI's MCPO proxy and point it at the ESPN MCP server.  Build the image
from the repository root:

```bash
docker build -t espn-mcp-backend .
```

The Docker image:

* Installs the package together with the optional `mcp` extra and the
  `open-webui-mcp` proxy runtime.
* Runs `mcpo-proxy serve` with defaults that bind to `0.0.0.0:${PORT}` (App
  Runner injects `PORT` at runtime) and spawns `python -m espn_api.mcp_server`
  behind the proxy.
* Lets you override the behaviour with environment variables:
  * `MCPO_PROXY_COMMAND` &mdash; if set, the container executes this shell command
    verbatim.  Use it when you want full control over the proxy invocation.
  * `MCPO_PROXY_HOST` &mdash; defaults to `0.0.0.0`.
  * `PORT` &mdash; defaults to `8000` for local testing but is automatically
    provided by App Runner.
  * `MCP_SERVER_COMMAND` &mdash; defaults to `python -m espn_api.mcp_server`; point
    it at a different MCP entrypoint if needed.

You can verify everything locally by running the container and curling the
OpenAPI description that the proxy exposes:

```bash
docker run --rm -p 8000:8000 espn-mcp-backend
curl http://localhost:8000/openapi.json | jq '.info.title'
```

Expect a title such as `"OpenWebUI MCP Proxy"`.  You can also use the proxy's
`/healthz` endpoint to check readiness.

## 3. Push the image to Amazon ECR

1. Create the repository (one time):
   ```bash
   aws ecr create-repository --repository-name espn-mcp-backend
   ```
2. Retrieve the login command and authenticate Docker:
   ```bash
   aws ecr get-login-password --region <region> \
     | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
   ```
3. Tag and push the image:
   ```bash
   docker tag espn-mcp-backend:latest <account-id>.dkr.ecr.<region>.amazonaws.com/espn-mcp-backend:latest
   docker push <account-id>.dkr.ecr.<region>.amazonaws.com/espn-mcp-backend:latest
   ```

## 4. Create the App Runner service

1. Open the **App Runner** console and choose **Create service**.
2. Select **Container registry** → **Amazon ECR** and pick the
   `espn-mcp-backend` repository.
3. Set **Deployment settings** to **Automatic** so App Runner redeploys on new
   image pushes.
4. Configure the service:
   * **Service name:** `espn-mcp-backend` (or similar).
   * **Port:** `8000` (matches the container's default `PORT`).
   * **Auto scaling:** keep the defaults (1 concurrent request per instance).
5. Add environment variables under **Service settings → Environment variables**:
   * `MCP_SERVER_COMMAND` (optional) if you need to pass extra CLI flags to the
     MCP server.  Leave unset to use the defaults.
   * `MCPO_PROXY_COMMAND` (optional) if you prefer to provide the entire proxy
     invocation yourself.
   * Secrets such as league credentials so tools can call private leagues.  Set
     `ESPN_MCP_SWID` and `ESPN_MCP_ESPN_S2` (or reference secrets with these
     names) to provide the cookies to the backend.
6. Attach an **Instance role** only if the backend needs AWS APIs (not required
   for the ESPN MCP server).
7. Review and create the service.  Initial provisioning takes a few minutes and
   results in a default HTTPS endpoint such as
   `https://<random>.<region>.awsapprunner.com`.

## 5. Share the backend URL with the frontend

Update the frontend so it targets the App Runner endpoint instead of
`http://localhost:8000`:

1. In the Amplify console open the app → **App settings → Environment
   variables**.
2. Add or update `VITE_BASE_URL` with the App Runner URL (or your custom
   domain).
3. Redeploy the Amplify app.  The frontend now proxies requests to the managed
   backend.

## 6. Secure and observe the backend

* **Restrict callers** by keeping the App Runner URL private and only surfacing
  it through the frontend.  If you need tighter controls, attach AWS WAF from
  the App Runner console.
* **Monitor** the built-in metrics (requests, CPU, memory, 4XX/5XX errors) from
  the service detail page.  Configure alerts or alarms in CloudWatch if needed.
* **Tag the service** (for example `Project=espn-mcp`) so you can track costs in
  the AWS Cost Explorer.

## 7. Optional: add a custom domain

1. In the App Runner console open the service → **Custom domains** →
   **Add domain**.
2. Enter a hostname such as `api.example.com` and follow the prompts.  App
   Runner provides CNAME records.
3. Create the DNS records in Route 53 (or your DNS provider) and wait for
   validation.
4. Once validated, update `VITE_BASE_URL` to reference the custom domain.

## 8. Wire CI/CD for future updates

Automate rebuilds and deployments by pushing through GitHub Actions (or another
CI system):

* Build the Docker image.
* Log in to ECR.
* Push the new tag (for example `latest` or a semantic version).

App Runner tracks the ECR repository; as soon as the new tag is available it
rolls out an updated service revision.

### Provide the ESPN cookies securely

App Runner encrypts environment variables at rest, but it is still best
practice to manage long-lived secrets outside of git and CI build logs.  Two
common patterns:

1. **Reference AWS Secrets Manager or Systems Manager Parameter Store.**
   When editing the App Runner service choose **Add environment variable →
   Reference existing secret** and pick the secret that stores the cookie value.
   Create secrets named `ESPN_MCP_SWID` and `ESPN_MCP_ESPN_S2` so the container
   receives the expected variables.
2. **Inject from CI/CD.**  If a GitHub Actions workflow deploys the service, add
   encrypted repository or organization secrets for `ESPN_MCP_SWID` and
   `ESPN_MCP_ESPN_S2`.  Pass them to the `aws apprunner update-service` command
   (or equivalent) using the `--environment-variables` flag.  Avoid baking the
   cookies into the Docker image or committing them to git.

With either approach the MCP server will automatically read the environment
variables and use them as defaults whenever a client request omits the cookies.

> **Why not store the cookies in the frontend?**  Amplify (and most static site
> hosts) bake `VITE_*` variables into the generated JavaScript bundle.  Anyone
> who loads the UI can read those values, so frontend environment variables do
> not keep the cookies private.  Keep the ESPN tokens on the backend and let the
> MCP server inject them automatically, or prompt individual users to paste
> their own cookies at runtime if you prefer not to store shared credentials.

---

With these pieces in place the ESPN Fantasy MCP backend runs as a fully managed
service, reachable via HTTPS, and ready for OpenWebUI or any other MCP-capable
client.

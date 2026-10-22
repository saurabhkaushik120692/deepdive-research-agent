# Deploying DeepDive to AWS (EC2 + ECR via GitHub Actions and SSM)

Pipeline: **push to `main` → GitHub Actions runs tests → builds the Docker image → pushes it to
ECR → tells the EC2 instance (through SSM, no SSH) to pull it and restart.** Caddy in front gives
free automatic HTTPS on a `nip.io` hostname.

The workflow only goes as far as the GitHub settings allow, so it's safe to push at any point:
tests always run; `build-push` starts once the `AWS_REGION` variable exists; `deploy` starts once
`EC2_INSTANCE_ID` exists too (step 7).

Region used below: **ap-southeast-2 (Sydney)**. Replace `<ACCOUNT_ID>` with your 12-digit account id
(top-right menu in the AWS console), `<GH_USER>` with `saurabhkaushik120692`, and `<INSTANCE_ID>`
with the instance id from step 5.

**How GitHub logs in to AWS.** On the AWS *free plan*, IAM identity providers (OIDC) are not
available, so GitHub uses the access key of a **deploy-only IAM user** whose permissions cover just
this ECR repository and sending commands to this one instance (steps 2–3). On a paid plan, use the
OIDC role instead ([Appendix](#appendix-oidc-role-paid-plan)): no stored keys at all. The workflow
supports both; it uses the role whenever the `AWS_ROLE_ARN` variable is set.

---

## 0. Account safety (done in Task 0)
Root MFA on, $10 budget alert on.

## 1. ECR repository
Console → **Elastic Container Registry → Create repository**
- Visibility: Private · Name: `deepdive-research-agent` · tag mutability: Mutable · rest default.
- (Optional, saves money) **Lifecycle policy**: expire untagged images after 1 day and keep the
  last 3 (the free plan includes 500 MB of private ECR storage; each image is ~400 MB compressed).

## 2. Deploy-only IAM user
Console → **IAM → Users → Create user**
- User name: `deepdive-github-deploy` · **leave "Provide user access to the AWS Management Console"
  unticked** (this user is for GitHub only, it never signs in).
- Permissions: **Attach policies directly** → don't tick anything → Next → **Create user**.

Open the user → **Permissions → Add permissions → Create inline policy → JSON**, paste, name it
`deepdive-deploy`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
    {"Effect": "Allow",
     "Action": ["ecr:BatchCheckLayerAvailability", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart",
                "ecr:CompleteLayerUpload", "ecr:PutImage", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
     "Resource": "arn:aws:ecr:ap-southeast-2:<ACCOUNT_ID>:repository/deepdive-research-agent"}
  ]
}
```

After step 5, edit this policy and add the two SSM statements below (sending commands to *only*
your instance, and reading their results):

```json
    {"Effect": "Allow", "Action": "ssm:SendCommand",
     "Resource": ["arn:aws:ec2:ap-southeast-2:<ACCOUNT_ID>:instance/<INSTANCE_ID>",
                  "arn:aws:ssm:ap-southeast-2::document/AWS-RunShellScript"]},
    {"Effect": "Allow", "Action": ["ssm:GetCommandInvocation", "ssm:ListCommandInvocations"], "Resource": "*"}
```

## 3. Access key → GitHub Secrets
User → **Security credentials → Access keys → Create access key** → use case **Third-party
service** (or "Application running outside AWS") → confirm → **Create access key**.

Copy both values **straight into GitHub** (they are shown only once; never paste them anywhere
else): GitHub → repo **Settings → Secrets and variables → Actions → Secrets → New repository secret**

| Secret | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | the access key id (`AKIA…`) |
| `AWS_SECRET_ACCESS_KEY` | the secret access key |

## 4. IAM role for the EC2 instance
Console → **IAM → Roles → Create role → AWS service → EC2**
- Attach managed policies: `AmazonSSMManagedInstanceCore`, `AmazonEC2ContainerRegistryReadOnly`
- Name: `deepdive-ec2`.

## 5. EC2 instance
Console → **EC2 → Launch instance**
- Name `deepdive` · AMI **Amazon Linux 2023** (x86_64) · type **t3.small** (on the free plan pick
  a type labelled *Free tier eligible*)
- Key pair: *Proceed without a key pair* (we use SSM, not SSH)
- Network: create a security group `deepdive-web` allowing **HTTP 80** and **HTTPS 443** from
  anywhere. **No SSH (22).**
- Storage: 20 GiB gp3
- Advanced details → IAM instance profile: `deepdive-ec2`
- Launch. Copy the **instance id** (`i-…`) and add the SSM statements to the policy from step 2.

## 6. Elastic IP + first-time setup on the instance
- **EC2 → Elastic IPs → Allocate**, then **Associate** with the `deepdive` instance.
  Your hostname is the IP with dashes + `.nip.io`, e.g. `13.201.5.7` → `13-201-5-7.nip.io`.
- **EC2 → instance → Connect → Session Manager → Connect** (browser shell, no SSH), then:

```bash
sudo dnf install -y git
git clone https://github.com/<GH_USER>/deepdive-research-agent.git   # private repo: use a fine-grained read-only token as the password
cd deepdive-research-agent && bash deploy/ec2-bootstrap.sh
sudo tee /opt/deepdive/.env > /dev/null <<'EOF'
MESH_API_KEY=rsk_...
TAVILY_API_KEY=tvly-...
GOOGLE_CLIENT_ID=...apps.googleusercontent.com   # step 9
GOOGLE_CLIENT_SECRET=GOCSPX-...
SESSION_SECRET=<python -c "import secrets; print(secrets.token_urlsafe(32))">
SITE_HOST=13-201-5-7.nip.io
RUNS_PER_HOUR=10
# image location, so a manual `docker compose up -d` on the instance works too
# (the deploy workflow sets these itself)
ECR_REGISTRY=<ACCOUNT_ID>.dkr.ecr.ap-southeast-2.amazonaws.com
ECR_REPOSITORY=deepdive-research-agent
EOF
sudo chmod 600 /opt/deepdive/.env
```

(If `deploy/docker-compose.prod.yml` or `deploy/Caddyfile` change later, `git pull` and re-run
the bootstrap script on the instance.)

## 7. GitHub repository variables
GitHub → repo **Settings → Secrets and variables → Actions → Variables** → add:

| Name | Value |
|---|---|
| `AWS_REGION` | `ap-southeast-2` |
| `ECR_REPOSITORY` | `deepdive-research-agent` |
| `EC2_INSTANCE_ID` | `i-…` |

These are variables, not secrets: none of them grants access on its own. Leave `AWS_ROLE_ARN`
unset on the free plan. (Setting `AWS_REGION` alone already lets a push build the image and push it
to ECR, which is a handy check before the instance exists.)

## 8. Deploy and check
Push to `main` (or Actions → deploy → Run workflow). Expected: `test`, `build-push`, `deploy` all
green, then:

```bash
curl -s https://13-201-5-7.nip.io/healthz     # {"status":"ok"}
```

Open the site, sign in with Google, run a question. Redeploy once more and check an earlier
run is still listed (the SQLite/Chroma data lives in `/opt/deepdive/data`, outside the container).

## 9. Google sign-in (OAuth client)
Starting a run needs a Google account. Create the OAuth client once; it serves both your laptop
and the live site.

1. <https://console.cloud.google.com> → project picker → **New project** → name `deepdive` → Create,
   and select it.
2. **APIs & Services → OAuth consent screen** (shown as **Google Auth Platform**) → **Get started**:
   app name `DeepDive`, your email as support and contact email, audience **External** → Create.
3. **Clients → Create client** → type **Web application**, name `DeepDive web`:
   - Authorized JavaScript origins: `http://localhost:8000`, `http://localhost:8001`,
     `https://<SITE_HOST>`
   - Authorized redirect URIs: `http://localhost:8000/auth/callback`,
     `http://localhost:8001/auth/callback`, `https://<SITE_HOST>/auth/callback`
   - **Create**, then copy the client id and client secret straight into `.env` (local) and
     `/opt/deepdive/.env` (server) as `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET`. The secret may
     be shown only once.
4. **Audience**: while the app is in *Testing*, only the test users you add there can sign in
   (add your own Google account). To let anyone sign in, click **Publish app**. The app only asks
   for name, email and profile picture (`openid email profile`), which needs no Google review.

Also set `SESSION_SECRET` (a random string, see the `.env` template above): it signs the login
cookie, so changing it signs everyone out. After editing the server's `.env`, restart the app:
`cd /opt/deepdive && sudo docker compose -f docker-compose.prod.yml up -d`.

If Google rejects the `nip.io` address in step 3, the site needs a real domain name (a few dollars
a year): point it at the Elastic IP, set `SITE_HOST` to it, and use it in step 3.

**Example reports** (public, listed on the home page for signed-out visitors) are runs you mark:

```bash
sudo docker compose -f docker-compose.prod.yml exec app python -m app.examples list
sudo docker compose -f docker-compose.prod.yml exec app python -m app.examples add <run_id>
```

## Costs and cleanup
On the free plan the eligible instance and ECR storage are covered by free-tier credits. An Elastic
IP is free while attached to a running instance. After results are announced: terminate the
instance, release the Elastic IP, **deactivate and delete the deploy user's access key** (IAM →
Users → `deepdive-github-deploy` → Security credentials) and delete the two GitHub secrets.

---

## Appendix: OIDC role (paid plan)
With IAM identity providers available, replace steps 2–3 with a role GitHub assumes per run, so no
key is stored anywhere:

1. **IAM → Identity providers → Add provider** → OpenID Connect · URL
   `https://token.actions.githubusercontent.com` · audience `sts.amazonaws.com`.
2. **IAM → Roles → Create role → Web identity** → that provider · organization `<GH_USER>` ·
   repository `deepdive-research-agent` · branch `main` · name `deepdive-github-deploy`. Its trust
   policy must restrict `token.actions.githubusercontent.com:sub` to
   `repo:<GH_USER>/deepdive-research-agent:ref:refs/heads/main`.
3. Attach the same inline policy as step 2 (ECR + SSM statements).
4. Set the variable `AWS_ROLE_ARN` = `arn:aws:iam::<ACCOUNT_ID>:role/deepdive-github-deploy` and
   delete the two access-key secrets.

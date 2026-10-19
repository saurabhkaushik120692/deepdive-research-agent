# Deploying DeepDive to AWS (EC2 + ECR, GitHub OIDC, SSM)

Pipeline: **push to `main` → GitHub Actions runs tests → builds the Docker image → pushes it to
ECR → tells the EC2 instance (through SSM, no SSH) to pull it and restart.** Caddy in front gives
free automatic HTTPS on a `nip.io` hostname.

Until the GitHub repository variables in step 7 exist, the `deploy` workflow only runs the tests
(build and deploy jobs are skipped), so it's safe to push before AWS is ready.

Region used below: **ap-southeast-2 (Sydney)**. Replace `<ACCOUNT_ID>` with your 12-digit account id
(top-right menu in the AWS console) and `<GH_USER>` with `saurabhkaushik120692`.

---

## 0. Account safety (done in Task 0)
Root MFA on, $10 budget alert on. Do the steps below as an IAM admin user, not root.

## 1. ECR repository
Console → **Elastic Container Registry → Create repository**
- Visibility: Private · Name: `deepdive-research-agent` · everything else default.
- (Optional, saves money) **Lifecycle policy**: expire untagged images, keep the last 3 (the free plan includes 500 MB of private ECR storage and each image is ~400 MB compressed).

## 2. GitHub OIDC identity provider (once per account)
Console → **IAM → Identity providers → Add provider**
- Type: OpenID Connect
- Provider URL: `https://token.actions.githubusercontent.com`
- Audience: `sts.amazonaws.com`

## 3. IAM role for GitHub Actions
Console → **IAM → Roles → Create role → Web identity**
- Identity provider: `token.actions.githubusercontent.com`, audience `sts.amazonaws.com`
- GitHub organization: `<GH_USER>` · repository: `deepdive-research-agent` · branch: `main`
- Skip the managed policies, name it `deepdive-github-deploy`, create.

Then open the role → **Trust relationships** and check it matches (only `main` of this repo):

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Federated": "arn:aws:iam::<ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com"},
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {
        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
        "token.actions.githubusercontent.com:sub": "repo:<GH_USER>/deepdive-research-agent:ref:refs/heads/main"
      }
    }
  }]
}
```

Add an **inline policy** (Permissions → Add permissions → Create inline policy → JSON). Fill in
`<INSTANCE_ID>` after step 5 (you can come back and edit it):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
    {"Effect": "Allow",
     "Action": ["ecr:BatchCheckLayerAvailability", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart",
                "ecr:CompleteLayerUpload", "ecr:PutImage", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
     "Resource": "arn:aws:ecr:ap-southeast-2:<ACCOUNT_ID>:repository/deepdive-research-agent"},
    {"Effect": "Allow", "Action": "ssm:SendCommand",
     "Resource": ["arn:aws:ec2:ap-southeast-2:<ACCOUNT_ID>:instance/<INSTANCE_ID>",
                  "arn:aws:ssm:ap-southeast-2::document/AWS-RunShellScript"]},
    {"Effect": "Allow", "Action": ["ssm:GetCommandInvocation", "ssm:ListCommandInvocations"], "Resource": "*"}
  ]
}
```

## 4. IAM role for the EC2 instance
Console → **IAM → Roles → Create role → AWS service → EC2**
- Attach managed policies: `AmazonSSMManagedInstanceCore`, `AmazonEC2ContainerRegistryReadOnly`
- Name: `deepdive-ec2`.

## 5. EC2 instance
Console → **EC2 → Launch instance**
- Name `deepdive` · AMI **Amazon Linux 2023** (x86_64) · type **t3.small**
- Key pair: *Proceed without a key pair* (we use SSM, not SSH)
- Network: create a security group `deepdive-web` allowing **HTTP 80** and **HTTPS 443** from
  anywhere. **No SSH (22).**
- Storage: 20 GiB gp3
- Advanced details → IAM instance profile: `deepdive-ec2`
- Launch. Copy the **instance id** (`i-…`) and put it into the policy from step 3.

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
DEMO_ACCESS_CODE=pick-a-code-for-judges
SITE_HOST=13-201-5-7.nip.io
RUNS_PER_HOUR=10
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
| `AWS_ROLE_ARN` | `arn:aws:iam::<ACCOUNT_ID>:role/deepdive-github-deploy` |
| `ECR_REPOSITORY` | `deepdive-research-agent` |
| `EC2_INSTANCE_ID` | `i-…` |

These are variables, not secrets: none of them grants access on its own. AWS keys never go into
GitHub; the workflow gets short-lived credentials through OIDC.

## 8. Deploy and check
Push to `main` (or Actions → deploy → Run workflow). Expected: `test`, `build-push`, `deploy` all
green, then:

```bash
curl -s https://13-201-5-7.nip.io/healthz     # {"status":"ok"}
```

Open the site, enter the access code, run a question. Redeploy once more and check an earlier
run is still listed (the SQLite/Chroma data lives in `/opt/deepdive/data`, outside the container).

## Costs and cleanup
t3.small ≈ $0.02/h (~$15/month) plus a few cents of ECR storage. An Elastic IP is free while
attached to a running instance. After results are announced: stop or terminate the instance and
release the Elastic IP.

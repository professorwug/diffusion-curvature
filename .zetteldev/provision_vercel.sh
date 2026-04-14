#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# provision_vercel_interactive.sh  —  One-shot Vercel project bootstrapper
# -----------------------------------------------------------------------------
# • Creates a static Vercel project via REST API
# • Links a GitHub repo
# • Sets buildCommand → node scripts/build-index.js and outputDirectory → .
# • Optionally binds a custom domain
# • Adds BASE_URL env‑var so Quarto templates work
# • Structured logging + error handling for easy troubleshooting
# -----------------------------------------------------------------------------
set -euo pipefail

plain()  { echo "[$(date +%T)] $*"; }
info()   { plain "ℹ️  $*"; }
success(){ plain "✅ $*"; }
error()  { plain "❌ $*" >&2; }
abort()  { error "$1"; exit 1; }

need_bin() { command -v "$1" &>/dev/null || abort "$1 is required"; }
for bin in curl jq; do need_bin "$bin"; done

read_secret() {
  local var="$1" prompt="$2" hidden="${3:-true}"
  if [[ -z "${!var-}" ]]; then
    if $hidden; then read -rsp "$prompt" "$var" && echo; else read -rp "$prompt" "$var"; fi
    export "$var"
  fi
}

api_call(){
  local m="$1" url="$2" data="${3:-}" body status
  body=$(curl -sS -X "$m" "${AUTH[@]}" -w "\n%{http_code}" ${data:+-d "$data"} "$url") || abort "Network error"
  status=${body##*$'\n'}; body=${body%$'\n'*}
  if [[ ! "$status" =~ ^20[01]$ ]]; then error "API $m $url → $status"; echo "$body"|jq . >&2; exit 1; fi
  echo "$body"
}

# ---------------- interactive prompts --------------------------------------- #
read_secret VERCEL_TOKEN "Enter Vercel API token (hidden): "
read_secret VERCEL_ORG_ID "Enter Vercel Team/Org ID (blank = personal): " false

read -rp "Project name: " PROJECT_NAME; [[ $PROJECT_NAME ]] || abort "Project name required"
read -rp "GitHub repo (user/repo): " REPO_SLUG;  [[ $REPO_SLUG ]]  || abort "Repo slug required"
read -rp "Root directory [.] : " ROOT_DIR; ROOT_DIR=${ROOT_DIR:-.}
read -rp "Custom domain (optional): " CUSTOM_DOMAIN

API="https://api.vercel.com"
AUTH=( -H "Authorization: Bearer $VERCEL_TOKEN" -H "Content-Type: application/json" )
TEAM_Q=${VERCEL_ORG_ID:+'?teamId='$VERCEL_ORG_ID}

# ---------------- 1. create project ----------------------------------------- #
info "Creating project $PROJECT_NAME …"
CREATE_PAYLOAD=$(jq -n --arg name "$PROJECT_NAME" --arg root "$ROOT_DIR" --arg slug "$REPO_SLUG" '{
  name:$name,
  framework:null,
  rootDirectory:$root,
  buildCommand:"node scripts/build-index.js",
  outputDirectory:".",
  gitRepository:{type:"github",repo:$slug,productionBranch:"main"}
}')

resp=$(api_call POST "$API/v9/projects$TEAM_Q" "$CREATE_PAYLOAD")
PROJECT_ID=$(jq -r '.id' <<<"$resp")
[[ $PROJECT_ID != null ]] || abort "Project id not returned"
success "Project created (id=$PROJECT_ID)"

# ---------------- 2. optional domain ---------------------------------------- #
if [[ $CUSTOM_DOMAIN ]]; then
  info "Binding domain $CUSTOM_DOMAIN …"
  api_call POST "$API/v9/projects/$PROJECT_ID/domains$TEAM_Q" "$(jq -n --arg name "$CUSTOM_DOMAIN" '{name:$name}')" >/dev/null
  PROD_URL="https://$CUSTOM_DOMAIN"
  success "Domain added"
else
  PROD_URL="https://$PROJECT_NAME.vercel.app"
fi

# ---------------- 3. add BASE_URL env var ----------------------------------- #
info "Adding BASE_URL env var …"
ENV_PAYLOAD=$(jq -n --arg key BASE_URL --arg val "$PROD_URL" '{key:$key,value:$val,target:["production"],type:"encrypted"}')
api_call POST "$API/v10/projects/$PROJECT_ID/env$TEAM_Q" "$ENV_PAYLOAD" >/dev/null
success "Environment variable set"

# ---------------- done ------------------------------------------------------- #
success "Provision complete! Site will deploy to $PROD_URL after first push."

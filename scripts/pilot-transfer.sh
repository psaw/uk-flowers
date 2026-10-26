#!/usr/bin/env bash
# Run only on the bridge host. All network connections originate here.
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
  echo "Usage: bash scripts/pilot-transfer.sh status | publish LOCAL_BRANCH CLOUD_BRANCH | return CLOUD_BRANCH sourcecraft/RETURN_BRANCH" >&2
  exit 2
}
validate_branch() { git check-ref-format "refs/heads/$1" >/dev/null; }
require_clean() {
  if [ -n "$(git status --porcelain)" ]; then
    echo "Commit or stash local changes before transferring branches." >&2
    exit 1
  fi
}
fetch_remotes() {
  git fetch onprem
  git fetch sourcecraft
}

[ "$#" -ge 1 ] || usage
action="$1"
case "$action" in
  status)
    [ "$#" -eq 1 ] || usage
    fetch_remotes
    git branch -avv
    ;;
  publish)
    [ "$#" -eq 3 ] || usage
    validate_branch "$2"
    validate_branch "$3"
    require_clean
    fetch_remotes
    git rev-parse --verify "refs/heads/$2^{commit}" >/dev/null
    git merge-base --is-ancestor refs/remotes/onprem/main "refs/heads/$2"
    git push sourcecraft "refs/heads/$2:refs/heads/$3"
    local_sha="$(git rev-parse "refs/heads/$2")"
    remote_sha="$(git ls-remote sourcecraft "refs/heads/$3" | cut -f1)"
    [ "$local_sha" = "$remote_sha" ]
    echo "Verified SourceCraft $3 at $remote_sha"
    ;;
  return)
    [ "$#" -eq 3 ] || usage
    validate_branch "$2"
    validate_branch "$3"
    case "$3" in sourcecraft/*) ;; *) echo "Return branch must start with sourcecraft/" >&2; exit 2 ;; esac
    require_clean
    fetch_remotes
    cloud_ref="refs/remotes/sourcecraft/$2"
    git rev-parse --verify "$cloud_ref^{commit}" >/dev/null
    git merge-base --is-ancestor refs/remotes/onprem/main "$cloud_ref"
    git push onprem "$cloud_ref:refs/heads/$3"
    cloud_sha="$(git rev-parse "$cloud_ref")"
    remote_sha="$(git ls-remote onprem "refs/heads/$3" | cut -f1)"
    [ "$cloud_sha" = "$remote_sha" ]
    echo "Verified onprem $3 at $remote_sha"
    ;;
  *) usage ;;
esac

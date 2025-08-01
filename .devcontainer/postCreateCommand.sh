
# 00 Configure apt-proxy
sed -i 's|archive.ubuntu.com/|host.docker.internal:8081/repository/apt-proxy-|g' /etc/apt/sources.list

# 50 Create shim for wget to rewrite urls
# cp shim-wget /usr/local/bin/wget
cat <<'EOF' > "/usr/local/bin/wget"
#!/bin/bash

# ==============================================================================
# Wget Shim Script
#
# This script intercepts calls to 'wget', rewrites specific GitHub URLs to
# a local repository, and then passes all arguments to the real wget command.
#
# Location: /usr/local/bin/wget
# ==============================================================================

# --- Configuration ---

# IMPORTANT: Path to the original wget binary. Verify this with 'which wget'.
REAL_WGET="/usr/bin/wget"

# Pattern 1: Matches GitHub release asset URLs.
# e.g., github.com/user/repo/releases/download/v1.0/asset.zip
URL_PATTERN_RELEASES="github.com/[^/]+/[^/]+/releases/download/[^/]+/.+"

# Pattern 2: Matches GitHub source code archive URLs (by tag or commit hash).
# e.g., github.com/user/repo/archive/refs/tags/v1.0.tar.gz
URL_PATTERN_ARCHIVES="github.com/[^/]+/[^/]+/archive/([0-9a-f]{40}|refs/tags/.+)\.(zip|tar\.gz)"

# The part of the URL to be replaced if a pattern matches.
STRING_TO_REPLACE="https://github.com/"
# The string to substitute in its place.
REPLACEMENT_STRING="http://host.docker.internal:8081/repository/github/"


# --- Sanity Check ---
# Prevent infinite loops if this script is somehow calling itself through the PATH.
if [ "$0" = "$REAL_WGET" ]; then
    echo "Wget shim error: REAL_WGET path points to itself. Aborting." >&2
    exit 1
fi


# --- Logic ---
declare -a new_args

for arg in "$@"; do
  # Check if the argument matches either of our defined URL patterns.
  if [[ "$arg" =~ $URL_PATTERN_RELEASES ]] || [[ "$arg" =~ $URL_PATTERN_ARCHIVES ]]; then
    # If it matches, perform the string replacement.
    rewritten_arg="${arg//$STRING_TO_REPLACE/$REPLACEMENT_STRING}"
    
    # Print a notification message to standard error.
    echo "wget shim: Rewriting GitHub URL: '$arg' -> '$rewritten_arg'" >&2
    
    # Add the rewritten argument to our list.
    new_args+=("$rewritten_arg")
  else
    # If it doesn't match, add the original argument.
    new_args+=("$arg")
  fi
done


# --- Execution ---
# Use 'exec' to replace this script process with the real wget command.
# This is more efficient than a direct call.
# All arguments, including the rewritten ones, are passed along correctly.
exec "$REAL_WGET" "${new_args[@]}"
EOF
chmod +x /usr/local/bin/wget

# 51 Create pip.conf
cat <<'EOF' > "/usr/pip.conf"
[global]
no-cache-dir = true

index = http://host.docker.internal:8081/repository/pypi-proxy/pypi
index-url = http://host.docker.internal:8081/repository/pypi-proxy/simple
extra-index-url =
        http://host.docker.internal:8081/repository/pypi-proxy-vsai-cuda-11/simple
        http://host.docker.internal:8081/repository/pypi-proxy-pytorch-cu118/simple
        https://pypi.ngc.nvidia.com
trusted-host =
        host.docker.internal
        pypi.ngc.nvidia.com
EOF


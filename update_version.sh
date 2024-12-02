#!/bin/bash
set -e

# Check if the file exists and is readable
if [ -f "$BINARY_VERSION_HASH" ]; then
    # Include (source) the file
    . "$BINARY_VERSION_HASH"
else
    echo "File not found: $BINARY_VERSION_HASH"
fi

if [[ "${SHOULD_BUILD}" != "yes" && "${FORCE_UPDATE}" != "true" ]]; then
  echo "Will not update version JSON because we did not build"
  exit 0
fi

if [[ -z "${GH_TOKEN}" ]] && [[ -z "${GITHUB_TOKEN}" ]] && [[ -z "${GH_ENTERPRISE_TOKEN}" ]] && [[ -z "${GITHUB_ENTERPRISE_TOKEN}" ]]; then
  echo "Will not update version JSON because no GITHUB_TOKEN defined"
  exit 0
else
  GITHUB_TOKEN="${GH_TOKEN:-${GITHUB_TOKEN:-${GH_ENTERPRISE_TOKEN:-${GITHUB_ENTERPRISE_TOKEN}}}}"
fi

# Support for GitHub Enterprise
GH_HOST="${GH_HOST:-github.com}"

if [[ "${FORCE_UPDATE}" == "true" ]]; then
  . version.sh
fi

## is build_sourceversion necessary
if [[ -z "${BUILD_SOURCEVERSION}" ]]; then
  echo "Will not update version JSON because no BUILD_SOURCEVERSION defined"
  exit 0
fi

REPOSITORY_NAME="${VERSIONS_REPOSITORY/*\//}"

generateJson() {
  local version_hash package_version timestamp
  JSON_DATA="{}"

  version_hash="${BINARY_VERSION_HASH}"
  package_version="${CARGO_PKG_VERSION}"
  timestamp=$( node -e 'console.log(Date.now())' )

  # check that nothing is blank (blank indicates something awry with build)
  for key in package_version version_hash; do
    if [[ -z "${key}" ]]; then
      echo "Variable '${key}' is empty; exiting..."
      exit 1
    fi
  done

  # generate json
  JSON_DATA=$( jq \
    --arg package_version "${package_version}" \
    --arg version_hash    "${version_hash}" \
    --arg timestamp       "${timestamp}" \
    '. | .version_hash=$version_hash | .package_version=$package_version | .timestamp=$timestamp' \
    <<<'{}' )
}

updateLatestVersion() {
  echo "Updating ${VERSION_PATH}/latest.json"

  # do not update the same version
  if [[ -f "${REPOSITORY_NAME}/${VERSION_PATH}/latest.json" ]]; then
    CURRENT_VERSION=$( jq -r '.name' "${REPOSITORY_NAME}/${VERSION_PATH}/latest.json" )
    echo "CURRENT_VERSION: ${CURRENT_VERSION}"

    if [[ "${CURRENT_VERSION}" == "${RELEASE_VERSION}" && "${FORCE_UPDATE}" != "true" ]]; then
      return 0
    fi
  fi

  echo "Generating ${VERSION_PATH}/latest.json"

  mkdir -p "${REPOSITORY_NAME}/${VERSION_PATH}"

  generateJson

  # prefixed with /extension – should update ide to have its own prefix
  # and fetch the correct version
  echo "${JSON_DATA}" > "${REPOSITORY_NAME}/extension/${VERSION_PATH}/latest.json"

  echo "${JSON_DATA}"
}


# init versions repo for later commiting + pushing the json file to it
# thank you https://www.vinaygopinath.me/blog/tech/commit-to-master-branch-on-github-using-travis-ci/
git clone "https://${GH_HOST}/${VERSIONS_REPOSITORY}.git"
cd "${REPOSITORY_NAME}" || { echo "'${REPOSITORY_NAME}' dir not found"; exit 1; }
git config user.email "$( echo "${GITHUB_USERNAME}" | awk '{print tolower($0)}' )-ci@not-real.com"
git config user.name "${GITHUB_USERNAME} CI"
git remote rm origin
git remote add origin "https://${GITHUB_USERNAME}:${GITHUB_TOKEN}@${GH_HOST}/${VERSIONS_REPOSITORY}.git" &> /dev/null

cd ../ # is this the right cd? ❓


# ❓ assets?

if [[ "${OS_NAME}" == "osx" ]]; then
  # ASSET_NAME="${APP_NAME}-darwin-${ARCH}-${RELEASE_VERSION}.zip"
  VERSION_PATH="${QUALITY}/darwin/${ARCH}"
  updateLatestVersion
elif [[ "${OS_NAME}" == "windows" ]]; then
  # system installer
  ASSET_NAME="${APP_NAME}Setup-${ARCH}-${RELEASE_VERSION}.exe"
  VERSION_PATH="${QUALITY}/win32/${ARCH}/system"
  updateLatestVersion

  # user installer
  ASSET_NAME="${APP_NAME}UserSetup-${ARCH}-${RELEASE_VERSION}.exe"
  VERSION_PATH="${QUALITY}/win32/${ARCH}/user"
  updateLatestVersion

  # windows archive
  ASSET_NAME="${APP_NAME}-win32-${ARCH}-${RELEASE_VERSION}.zip"
  VERSION_PATH="${QUALITY}/win32/${ARCH}/archive"
  updateLatestVersion

  if [[ "${ARCH}" == "ia32" || "${ARCH}" == "x64" ]]; then

    # ❓ assets?

    # ❓ msi or non msi?
    # ASSET_NAME="${APP_NAME}-${ARCH}-${RELEASE_VERSION}.msi"
    VERSION_PATH="${QUALITY}/win32/${ARCH}/msi"
    updateLatestVersion

    # updates-disabled msi
    ASSET_NAME="${APP_NAME}-${ARCH}-updates-disabled-${RELEASE_VERSION}.msi"
    VERSION_PATH="${QUALITY}/win32/${ARCH}/msi-updates-disabled"
    updateLatestVersion
  fi
else # linux
  # update service links to tar.gz file
  # see https://update.code.visualstudio.com/api/update/linux-x64/stable/VERSION
  # as examples
  ASSET_NAME="${APP_NAME}-linux-${ARCH}-${RELEASE_VERSION}.tar.gz"
  VERSION_PATH="${QUALITY}/linux/${ARCH}"
  updateLatestVersion
fi

cd "${REPOSITORY_NAME}" || { echo "'${REPOSITORY_NAME}' dir not found"; exit 1; }

git pull origin main # in case another build just pushed
git add .

CHANGES=$( git status --porcelain )

if [[ -n "${CHANGES}" ]]; then
  echo "Some changes have been found, pushing them"

  dateAndMonth=$( date "+%D %T" )

  git commit -m "CI update: ${dateAndMonth} (Build ${GITHUB_RUN_NUMBER})"

  if ! git push origin main --quiet; then
    git pull origin main
    git push origin main --quiet
  fi
else
  echo "No changes"
fi

# cd .. is this important?
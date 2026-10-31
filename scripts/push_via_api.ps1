<#
.SYNOPSIS
    Upload the tracked files to a GitHub repository through the REST API.

.DESCRIPTION
    `git push` over HTTPS is unreliable from this machine -- the connection is
    cut mid-transfer ("curl 52 Empty reply from server").  The REST API answers
    fine, so we upload file by file with the contents API.

    Two machine-specific rules are baked in here:

    * The campus network cuts a connection that has no answer for about twenty
      seconds, so every request has to be small and the API has to answer
      quickly.  Files are therefore pushed one at a time.
    * Files that disappeared from the repository are deleted from the remote
      too, so the remote never keeps a stale copy of a file we replaced.

.PARAMETER Repo
    owner/name of the repository.

.EXAMPLE
    pwsh scripts/push_via_api.ps1 -Repo Microbiosis/seance
#>
param(
    [string]$Repo = 'Microbiosis/seance',
    [string]$Branch = 'main',
    [string]$Message = 'seance: update',
    [int]$Tries = 6
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

$token = (& gh auth token).Trim()
if (-not $token) { throw 'no GitHub token: run `gh auth login`' }

$headers = @{
    Authorization        = "Bearer $token"
    Accept               = 'application/vnd.github+json'
    'X-GitHub-Api-Version' = '2022-11-28'
    'User-Agent'         = 'seance-push'
}

$files = (& git -C $root ls-files) | Where-Object { $_ }
# the figures and the README first: the DEV post links them, so they want to
# be on the remote before the long tail of asset parts
$priority = @('README.md', 'results/')
$files = @($files | Where-Object { $p = $_; ($priority | Where-Object { $p -like "$_*" }).Count -gt 0 }) +
         @($files | Where-Object { $p = $_; ($priority | Where-Object { $p -like "$_*" }).Count -eq 0 })
Write-Output "[push] $($files.Count) files -> $Repo"

# what is already on the remote?
$remote = @{}
try {
    $tree = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/git/trees/$Branch`?recursive=1" `
            -Headers $headers -TimeoutSec 30
    foreach ($n in $tree.tree) { if ($n.type -eq 'blob') { $remote[$n.path] = $n.sha } }
} catch {
    Write-Output "[push] could not read the remote tree: $($_.Exception.Message)"
}

$ok = 0
$failed = @()
$deleted = 0
foreach ($path in $files) {
    $full = Join-Path $root $path
    if (-not (Test-Path -LiteralPath $full)) { continue }
    $b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($full))
    $url = "https://api.github.com/repos/$Repo/contents/$path"

    # updating a file that already exists needs the sha of the remote copy
    $payload = @{ message = "update $path"; content = $b64; branch = $Branch }
    if ($remote.ContainsKey($path)) { $payload['sha'] = $remote[$path] }
    $json = $payload | ConvertTo-Json -Depth 3

    $done = $false
    for ($i = 1; $i -le $Tries -and -not $done; $i++) {
        try {
            $r = Invoke-RestMethod -Uri $url -Method Put -Headers $headers -Body $json -ContentType 'application/json' -TimeoutSec 45
            $ok++
            Write-Output "[push]   ok $path"
            $done = $true
        }
        catch {
            $msg = $_.Exception.Message
            $detail = $_.ErrorDetails.Message
            if ($i -eq $Tries) {
                Write-Output "[push]   FAILED $path : $msg $detail"
                $failed += $path
            }
            else { Start-Sleep -Seconds (3 * $i) }
        }
    }
    # this link throttles bursts of new connections: keep a slow, steady pace
    Start-Sleep -Seconds 3
}

# files that are on the remote but no longer in the repository
foreach ($path in @($remote.Keys)) {
    if ($files -contains $path) { continue }
    if ($path -in @('.gitattributes')) { continue }
    try {
        $body = @{ message = "remove $path"; sha = $remote[$path]; branch = $Branch } | ConvertTo-Json -Depth 3
        Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/contents/$path" -Method Delete `
            -Headers $headers -Body $body -ContentType 'application/json' -TimeoutSec 30 | Out-Null
        $deleted++
        Write-Output "[push]   removed $path"
    } catch {
        Write-Output "[push]   could not remove $path : $($_.Exception.Message)"
    }
}

Write-Output "[push] uploaded $ok/$($files.Count), removed $deleted"
if ($failed.Count) { Write-Output "[push] failed: $($failed -join ', ')" }

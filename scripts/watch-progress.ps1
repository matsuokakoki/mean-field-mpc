param(
    [switch]$FollowLog
)

$root = Split-Path -Parent $PSScriptRoot
$progress = Join-Path $root 'artifacts\progress.json'
Get-Content -Raw -LiteralPath $progress

if ($FollowLog) {
    $latest = Get-ChildItem -LiteralPath (Join-Path $root 'logs') -Filter 'reproduce_*.log' |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $latest) { throw 'No reproduce log exists yet.' }
    Get-Content -LiteralPath $latest.FullName -Wait
}

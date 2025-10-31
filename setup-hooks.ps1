$repoRoot = (git rev-parse --show-toplevel)
$hooksDir = Join-Path $repoRoot ".githooks"
$gitHooksDir = Join-Path $repoRoot ".git/hooks"

Write-Host "Installing Git hooks..."

New-Item -ItemType Directory -Force -Path $gitHooksDir | Out-Null

Get-ChildItem $hooksDir | ForEach-Object {
    $hookName = $_.Name
    $src = $_.FullName
    $dst = Join-Path $gitHooksDir $hookName

    Copy-Item $src $dst -Force
    Write-Host "Installed $hookName"
}

Write-Host "Git hooks installation complete!"

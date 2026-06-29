{
  pkgs,
  treefmt-nix,
  projectRootFile ? "flake.nix",
}:

treefmt-nix.lib.evalModule pkgs (import ./formatter-module.nix { inherit projectRootFile; })

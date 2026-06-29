{
  projectRootFile ? "flake.nix",
}:

{
  inherit projectRootFile;

  programs.rustfmt.enable = true;
  programs.nixfmt.enable = true;
  programs.taplo.enable = true;
}

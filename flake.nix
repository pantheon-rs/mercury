{
  description = "Mercury: generic differentiable math substrate for pantheon-rs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    treefmt-nix.url = "github:numtide/treefmt-nix";
    crane.url = "github:ipetkov/crane";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      treefmt-nix,
      crane,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
        craneLib = crane.mkLib pkgs;
        formatter = import ./nix/formatter.nix {
          inherit pkgs treefmt-nix;
          projectRootFile = "flake.nix";
        };
        packages = import ./nix/packages.nix {
          inherit pkgs craneLib;
          src = ./.;
        };
        checks = import ./nix/checks.nix {
          inherit
            self
            pkgs
            craneLib
            formatter
            ;
          inherit (packages) commonArgs cargoArtifacts mercury;
        };
        devShells = import ./nix/dev-shells.nix {
          inherit pkgs formatter;
        };
      in
      {
        packages = {
          default = packages.mercury;
          mercury = packages.mercury;
        };

        checks = checks;
        devShells = devShells;
        formatter = formatter.config.build.wrapper;
      }
    );
}

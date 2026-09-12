{
  description = "Mercury: generic differentiable math substrate for pantheon-rs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    treefmt-nix.url = "github:numtide/treefmt-nix";
    crane.url = "github:ipetkov/crane";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      treefmt-nix,
      crane,
      rust-overlay,
    }:
    # Only this platform is validated by Mercury's Enzyme checks.
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
        };
        rustWithEnzyme = import ./nix/rust-toolchain.nix { inherit pkgs; };
        # Use the same pinned compiler for developer runs and sandboxed checks.
        craneLib = (crane.mkLib pkgs).overrideToolchain rustWithEnzyme;
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
          inherit pkgs formatter rustWithEnzyme;
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

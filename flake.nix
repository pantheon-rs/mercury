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
    # Restricted to the one system the pinned Enzyme artifact exists for
    # (nix/rust-toolchain.nix fetches a prebuilt x86_64-linux libEnzyme; other
    # systems would get a toolchain with a broken sysroot).
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
        };
        rustWithEnzyme = import ./nix/rust-toolchain.nix { inherit pkgs; };
        # The checks must compile with the SAME pinned nightly+Enzyme
        # toolchain as the dev shell: the crate is nightly-only and its
        # tests exercise -Zautodiff codegen.
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

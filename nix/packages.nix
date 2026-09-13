{
  pkgs,
  craneLib,
  src,
}:

let
  cargoToml = builtins.fromTOML (builtins.readFile ../Cargo.toml);

  # Public API guides are included in the crate's rustdoc and executed as doctests.
  filteredSrc = pkgs.lib.cleanSourceWith {
    inherit src;
    filter = path: type: (craneLib.filterCargoSources path type) || pkgs.lib.hasSuffix ".md" path;
  };

  commonArgs = {
    pname = cargoToml.package.name;
    version = cargoToml.package.version;
    src = filteredSrc;
    strictDeps = true;
    # Enzyme's derivative pass: mandatory for compiling #[autodiff_reverse]
    # code. Fat LTO comes from Cargo.toml's release profile; crane builds
    # release by default.
    RUSTFLAGS = "-Zautodiff=Enable";
    RUSTDOCFLAGS = "-Zautodiff=Enable -Clto=fat -Ccodegen-units=1 -Copt-level=3";
  };

  cargoArtifacts = craneLib.buildDepsOnly commonArgs;

  mercury = craneLib.buildPackage (
    commonArgs
    // {
      inherit cargoArtifacts;
      doCheck = true;
      cargoExtraArgs = "--workspace --all-features";
    }
  );
in
{
  inherit commonArgs cargoArtifacts mercury;
}

{
  pkgs,
  craneLib,
  src,
}:

let
  cargoToml = builtins.fromTOML (builtins.readFile ../Cargo.toml);

  filteredSrc = craneLib.cleanCargoSource src;

  commonArgs = {
    pname = cargoToml.package.name;
    version = cargoToml.package.version;
    src = filteredSrc;
    strictDeps = true;
    # Enzyme's derivative pass: mandatory for compiling #[autodiff_reverse]
    # code. Fat LTO comes from Cargo.toml's release profile; crane builds
    # release by default.
    RUSTFLAGS = "-Zautodiff=Enable";
  };

  cargoArtifacts = craneLib.buildDepsOnly commonArgs;

  mercury = craneLib.buildPackage (
    commonArgs
    // {
      inherit cargoArtifacts;
      doCheck = true;
      cargoExtraArgs = "--all-features";
    }
  );
in
{
  inherit commonArgs cargoArtifacts mercury;
}

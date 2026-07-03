{
  self,
  pkgs,
  craneLib,
  formatter,
  commonArgs,
  cargoArtifacts,
  mercury,
}:

let
  withArtifacts = commonArgs // {
    inherit cargoArtifacts;
  };
in
{
  package = mercury;

  formatting = formatter.config.build.check self;

  clippy = craneLib.cargoClippy (
    withArtifacts
    // {
      cargoClippyExtraArgs = "--all-targets --all-features -- -D warnings";
    }
  );

  tests = craneLib.cargoTest (
    withArtifacts
    // {
      cargoExtraArgs = "--all-features";
    }
  );

  docs = craneLib.cargoDoc (
    withArtifacts
    // {
      cargoDocExtraArgs = "--no-deps --all-features";
    }
  );

  # Build every example and run the Phase 2 demo end-to-end (metis parity:
  # examples are part of CI, not decoration).
  examples = craneLib.mkCargoDerivation (
    withArtifacts
    // {
      pnameSuffix = "-examples";
      buildPhaseCargoCommand = ''
        cargo build --release --locked --examples
        cargo run --release --locked --example solve_gradient
      '';
      doCheck = false;
    }
  );
}

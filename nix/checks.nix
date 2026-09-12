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
      cargoClippyExtraArgs = "--workspace --all-targets --all-features -- -D warnings";
    }
  );

  tests = craneLib.cargoTest (
    withArtifacts
    // {
      cargoExtraArgs = "--workspace --all-features";
    }
  );

  docs = craneLib.cargoDoc (
    withArtifacts
    // {
      cargoDocExtraArgs = "--workspace --no-deps --all-features";
    }
  );

}

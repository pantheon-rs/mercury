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
}

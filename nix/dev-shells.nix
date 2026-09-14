{
  pkgs,
  formatter,
  rustWithEnzyme,
}:

{
  # Shared environment for compiler smoke tests and project checks.
  default = pkgs.mkShell {
    packages = [
      rustWithEnzyme
    ]
    ++ (with pkgs; [
      git
      jq
      just
      pkg-config
      formatter.config.build.wrapper
      rust-analyzer
      cargo-deny
      cargo-semver-checks
    ]);

    RUSTFLAGS = "-Zautodiff=Enable";
    # Rustdoc compiles examples separately from Cargo's release profile.
    RUSTDOCFLAGS = "-Zautodiff=Enable -Clto=fat -Ccodegen-units=1 -Copt-level=3";
    MERCURY_ENZYME_SHELL = "1";

    shellHook = ''
      export RUST_BACKTRACE=1
      echo "Mercury Enzyme shell"
      echo "  rustc: $(rustc --version 2>/dev/null)"
    '';
  };
}

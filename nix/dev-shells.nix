{
  pkgs,
  formatter,
  rustWithEnzyme,
}:

let
  llvmTools = pkgs.llvmPackages.llvm;
in
{
  # The one development shell. Mercury is nightly-only (#![feature(autodiff)])
  # and every derivative requires the pinned Enzyme toolchain, so there is
  # exactly one environment that can build it — this one. Profiling shells
  # can return as a separate entry when there is something to profile.
  default = pkgs.mkShell {
    packages = [
      rustWithEnzyme
      llvmTools
    ]
    ++ (with pkgs; [
      git
      jq
      just
      pkg-config
      formatter.config.build.wrapper
      rust-analyzer
      cargo-deny
      cargo-llvm-cov
      cargo-semver-checks
    ]);

    RUSTFLAGS = "-Zautodiff=Enable";
    MERCURY_ENZYME_SHELL = "1";

    shellHook = ''
      export RUST_BACKTRACE=1
      export LLVM_COV=${llvmTools}/bin/llvm-cov
      export LLVM_PROFDATA=${llvmTools}/bin/llvm-profdata
      echo "Mercury Enzyme shell"
      echo "  rustc: $(rustc --version 2>/dev/null)"
    '';
  };
}

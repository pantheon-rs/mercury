{
  pkgs,
  formatter,
  rustWithEnzyme,
}:

let
  llvmTools = pkgs.llvmPackages.llvm;
  llvmCoverageEnv = ''
    export LLVM_COV=${llvmTools}/bin/llvm-cov
    export LLVM_PROFDATA=${llvmTools}/bin/llvm-profdata
  '';

  commonTools = with pkgs; [
    git
    jq
    just
    pkg-config
    formatter.config.build.wrapper
  ];

  enzymeTools = [
    rustWithEnzyme
    llvmTools
  ]
  ++ commonTools
  ++ (with pkgs; [
    rust-analyzer
    cargo-deny
    cargo-llvm-cov
    cargo-semver-checks
  ]);

  enzymeShell = pkgs.mkShell {
    packages = enzymeTools;

    RUSTFLAGS = "-Zautodiff=Enable";
    MERCURY_ENZYME_SHELL = "1";

    shellHook = ''
      export RUST_BACKTRACE=1
      ${llvmCoverageEnv}
      echo "Mercury Enzyme shell"
      echo "  rustc: $(rustc --version 2>/dev/null)"
    '';
  };
in
{
  default = enzymeShell;

  bootstrap = pkgs.mkShell {
    packages = with pkgs; [
      cargo
      rustc
      git
      llvmTools
      formatter.config.build.wrapper
    ];

    shellHook = llvmCoverageEnv;
  };

  perf = pkgs.mkShell {
    packages =
      enzymeTools
      ++ (with pkgs; [
        cargo-flamegraph
        heaptrack
        valgrind
      ])
      ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
        pkgs.perf
      ];

    shellHook = ''
      export RUST_BACKTRACE=1
      ${llvmCoverageEnv}
      echo "Mercury performance shell"
    '';
  };

  enzyme = enzymeShell;
}

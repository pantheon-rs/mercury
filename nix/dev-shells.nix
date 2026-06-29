{
  pkgs,
  formatter,
}:

let
  llvmTools = pkgs.llvmPackages.llvm;
  llvmCoverageEnv = ''
    export LLVM_COV=${llvmTools}/bin/llvm-cov
    export LLVM_PROFDATA=${llvmTools}/bin/llvm-profdata
  '';

  rustTools = with pkgs; [
    cargo
    rustc
    rustfmt
    clippy
    rust-analyzer
    cargo-nextest
    cargo-llvm-cov
    cargo-deny
    cargo-semver-checks
    llvmTools
  ];

  commonTools = with pkgs; [
    git
    jq
    just
    pkg-config
    formatter.config.build.wrapper
  ];
in
{
  default = pkgs.mkShell {
    packages = rustTools ++ commonTools;

    shellHook = ''
      export RUST_BACKTRACE=1
      ${llvmCoverageEnv}
      echo "Mercury development shell"
    '';
  };

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
      rustTools
      ++ commonTools
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
}

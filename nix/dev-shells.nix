{
  pkgs,
  formatter,
}:

let
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
      echo "Mercury development shell"
    '';
  };

  bootstrap = pkgs.mkShell {
    packages = with pkgs; [
      cargo
      rustc
      git
      formatter.config.build.wrapper
    ];
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
      echo "Mercury performance shell"
    '';
  };
}

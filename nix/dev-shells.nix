{
  pkgs,
  formatter,
}:

let
  enzymeTarget = "x86_64-unknown-linux-gnu";
  enzymeLib = pkgs.fetchzip {
    url = "https://ci-artifacts.rust-lang.org/rustc-builds/ec818fda361ca216eb186f5cf45131bd9c776bb4/enzyme-nightly-x86_64-unknown-linux-gnu.tar.xz";
    sha256 = "sha256-Rnrop44vzS+qmYNaRoMNNMFyAc3YsMnwdNGYMXpZ5VY=";
  };
  llvmTools = pkgs.llvmPackages.llvm;
  llvmCoverageEnv = ''
    export LLVM_COV=${llvmTools}/bin/llvm-cov
    export LLVM_PROFDATA=${llvmTools}/bin/llvm-profdata
  '';
  rustWithEnzyme = pkgs.symlinkJoin {
    name = "rust-with-enzyme";
    paths = [ pkgs.rust-bin.nightly."2026-03-03".default ];
    nativeBuildInputs = [ pkgs.makeWrapper ];
    postBuild = ''
      libdir=$out/lib/rustlib/${enzymeTarget}/lib
      cp ${enzymeLib}/enzyme-preview/lib/rustlib/${enzymeTarget}/lib/libEnzyme-22.so "$libdir/"
      for tool in rustc rustdoc clippy-driver; do
        if [ -x "$out/bin/$tool" ]; then
          wrapProgram "$out/bin/$tool" --add-flags "--sysroot $out"
        fi
      done
    '';
  };

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

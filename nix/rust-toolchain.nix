# The pinned nightly + Enzyme toolchain — the single definition shared by the
# dev shells AND the flake checks (crane). Everything that compiles mercury
# must use this toolchain: the crate is nightly-only (#![feature(autodiff)])
# and derivatives require the libEnzyme plugin in the sysroot.
#
# Pin rationale: libEnzyme-22.so <=> LLVM 22 <=> the nightly merged 2026-03-02
# (commit ec818fda). nightly."2026-03-03" resolves to exactly that commit;
# newer nightlies have drifted past LLVM 22 and refuse to load the plugin.
{ pkgs }:

let
  enzymeTarget = "x86_64-unknown-linux-gnu";
  enzymeLib = pkgs.fetchzip {
    url = "https://ci-artifacts.rust-lang.org/rustc-builds/ec818fda361ca216eb186f5cf45131bd9c776bb4/enzyme-nightly-x86_64-unknown-linux-gnu.tar.xz";
    sha256 = "sha256-Rnrop44vzS+qmYNaRoMNNMFyAc3YsMnwdNGYMXpZ5VY=";
  };
in
pkgs.symlinkJoin {
  name = "rust-with-enzyme";
  paths = [ pkgs.rust-bin.nightly."2026-03-03".default ];
  nativeBuildInputs = [ pkgs.makeWrapper ];
  postBuild = ''
    libdir=$out/lib/rustlib/${enzymeTarget}/lib
    cp ${enzymeLib}/enzyme-preview/lib/rustlib/${enzymeTarget}/lib/libEnzyme-22.so "$libdir/"
    # symlinkJoin means binaries self-locate their sysroot at the ORIGINAL
    # rust-bin store path (no libEnzyme there); pin it explicitly per tool.
    for tool in rustc rustdoc; do
      if [ -x "$out/bin/$tool" ]; then
        wrapProgram "$out/bin/$tool" --add-flags "--sysroot $out"
      fi
    done
    # clippy-driver detects cargo's wrapper protocol by checking that its
    # FIRST argument is rustc — so the sysroot flag must be APPENDED, not
    # prepended, or it treats "rustc" and "-" as two input files.
    if [ -x "$out/bin/clippy-driver" ]; then
      wrapProgram "$out/bin/clippy-driver" --append-flags "--sysroot $out"
    fi
    # cargo-clippy locates clippy-driver next to its own REALPATH
    # (current_exe), which through a symlinkJoin resolves to the original
    # unwrapped store dir — bypassing the wrapper and failing the Enzyme
    # sysroot probe. A real copy keeps the sibling lookup inside $out.
    if [ -e "$out/bin/cargo-clippy" ]; then
      real=$(readlink -f "$out/bin/cargo-clippy")
      rm "$out/bin/cargo-clippy"
      cp "$real" "$out/bin/cargo-clippy"
    fi
  '';
}

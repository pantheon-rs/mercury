# One pinned compiler and matching distributed Enzyme component for builds,
# tests, clippy and documentation. The March CI artifact is no longer available.
{ pkgs }:

pkgs.rust-bin.nightly."2026-06-23".default.override {
  extensions = [ "enzyme" ];
}

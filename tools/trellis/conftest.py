"""pytest: test_tqenc_py.py here is a script, not a pytest module. It runs its checks when it is imported and needs
libtqenc.so, which tools/trellis/build.sh makes on a box with a compiler; collecting it would stop the whole pytest run.
Run it as  python3 tools/trellis/test_tqenc_py.py  on that box."""
collect_ignore_glob = ["test_*.py"]

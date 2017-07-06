for f in $(find '.' -mindepth 1 -maxdepth 1 -type d ); do
    compress=(tar cJvf $f{.tar.xz,})
    remove=(rm -rf $f)
    ${compress[@]} && ${remove[@]} &
done

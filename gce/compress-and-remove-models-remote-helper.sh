cd /home/$USER/data/*/*/*/models
for f in $(find '.' -mindepth 1 -maxdepth 1 -type d ); do
    compress=(sudo tar cJvf $f{.tar.xz,})
    remove=(sudo rm -rf $f)
    ${compress[@]} && ${remove[@]}
done

# Original notices

The complete source archive preserves the upstream notices and license texts. Copies here identify
the runtime dependencies and benchmark sources. Original notice text is preserved; its references
to source files are relative to the corresponding complete source tree.

The native bridge uses the original TensorFold RDMA path; SparkComm is disabled in this recipe.
The `borrowed-native-bridge-source.tar` release asset includes the historical CPU build
record and its limits. The exact external gather-module producer record has not been located;
related source and its original notice are supplied in `glm53-external-module-sources.tar.gz`.

The portable runtime image retains the installed distribution packages' copyright and license
files. Matching base-image and overlay source-package versions are supplied in the corresponding
source archive, including both versions where the overlay differs from base package metadata.
Model-provider licenses are separate; weights are not redistributed in this recipe.

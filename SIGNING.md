# Release signatures

This release uses the temporary Ed25519 signing key explicitly selected by the maintainer.
The author and committer are Anton Starikov
`<2550573+antst@users.noreply.github.com>`. Signing uses process-local Git settings;
the maintainer's normal signing configuration is unchanged.

Public-key fingerprint:

```text
SHA256:Cvk54dwEpcJjiuIiqlXrwstNKnXtDMh8tXDKpdD+LD4
```

`SIGNING-KEY.pub` contains the public key. `ALLOWED_SIGNERS` permits that key for
Git signatures under the stated maintainer identity. Neither file contains a private key.

From a clone containing the source branches, verify a snapshot listed in
[TP4 source identities](SOURCE.json) or [TP2 source identities](tp2/SOURCE.json).
This example checks the main TP4 engine:

```sh
ssh-keygen -lf SIGNING-KEY.pub
git -c gpg.format=ssh -c gpg.ssh.program=ssh-keygen \
  -c gpg.ssh.allowedSignersFile="$PWD/ALLOWED_SIGNERS" \
  verify-commit 1e3ba2cf32cb1dea035e38ad41adffc7ea402c6d
```

The same settings work with `verify-tag` for the release tag. Complete source archives
retain Git's commit identifier in their PAX metadata, and every source-file SHA256 was
compared with its frozen build manifest before the source snapshot was signed.
The snapshots were created after the measured builds; signatures attest the recorded
source bytes, without claiming a new bit-identical build or changing the tested executables.

GitHub's account-level signature recognition is separate from this cryptographic check.
This temporary public key was not on the account's SSH signing-key list when the release
was prepared. The release therefore claims verification against the supplied public key,
not a GitHub “Verified” badge. Account signing-key settings were not changed.

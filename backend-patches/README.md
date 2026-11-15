# Physical AI Studio integration patch

`manual-jog.patch` captures the station's manual-jog backend changes, including new source and tests, against the pinned upstream submodule. Keeping the changes as a patch preserves upstream history and licensing.

From a clean checkout at the recorded submodule commit:

```sh
git -C vendor/physical-ai-studio apply --check ../../backend-patches/manual-jog.patch
git -C vendor/physical-ai-studio apply ../../backend-patches/manual-jog.patch
```

Do not apply twice or overwrite a running installation. The current development submodule may already contain this patch. The dashboard uses the corresponding command schema. Follow upstream backend setup and test instructions before deployment; patch application itself does not restart the backend.

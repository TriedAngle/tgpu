# Source this from fish:   source scripts/moltenvk.fish
#
# Points the Vulkan loader at the newest installed LunarG SDK on macOS.
#
# Picking the *newest* SDK matters: older SDKs bundle old MoltenVK. On this
# machine 1.4.309/1.4.313 ship MoltenVK with ICD api_version 1.2.0, while
# 1.4.341 ships MoltenVK 1.4 (api_version 1.4.0) plus KosmicKrisp (1.3.0).

set -l sdk_root $HOME/VulkanSDK
set -q VULKAN_SDK_ROOT; and set sdk_root $VULKAN_SDK_ROOT

set -l sdk (ls -d $sdk_root/*/macOS 2>/dev/null | sort -V | tail -n 1)

if test -z "$sdk"
    echo "moltenvk.fish: no Vulkan SDK found under $sdk_root" >&2
    return 1
end

set -gx VULKAN_SDK $sdk
set -gx DYLD_FALLBACK_LIBRARY_PATH "$sdk/lib"
set -gx VK_ICD_FILENAMES "$sdk/share/vulkan/icd.d/MoltenVK_icd.json"
set -gx VK_LAYER_PATH "$sdk/share/vulkan/explicit_layer.d"

set -l api_version unknown
if command -sq python3
    set api_version (python3 -c "import json;print(json.load(open('$VK_ICD_FILENAMES'))['ICD']['api_version'])" 2>/dev/null)
end

echo "Vulkan SDK: $VULKAN_SDK ($api_version ICD)" >&2

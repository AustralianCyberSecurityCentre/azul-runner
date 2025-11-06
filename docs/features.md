# Features

The core of Azul processing is the ability to extract normalised features from entities.
This allows correlation between entity instances to find and cluster related samples.

Features are simple name, value pairs. They have additional metadata to define what data
type they are (for validation and improved decomposition/indexing) and values can include
details such as the offset and size of where the feature occurs in an entity's data stream.

It is dependent on plugin authors to choose good feature names and values to extract.
While there are always exceptions, the following will provide general guidance on what to
consider when defining features.

## What to Feature

Features have two primary purposes:

They can be as a mechanism to inform analysts of something interesting or notable about an
entity. For example that a binary is of a known malware family.

Secondly, they are used as a means to correlate and pivot to other entities that share the same
feature value. For example the hash of a shared child resource.

Most features fall into both categories, whereas, some are more heavily skewed toward
informational or for correlation.

## Feature Names

The following are some guidelines on naming features and some common pitfalls.

### Names should be singular (non-plural)

Even if samples will contain more than one value of the feature the name should be singular.

- eg. `pe_export_function` not `pe_export_functions`

### Names should describe the technique or data being extracted

Feature names shouldn't refer to the tool used to perform the extraction. This is a departure
from Azul 2 where non-authored features were named after the plugin. In Azul 3 all features
are authored/reusable between plugins.

Including a short file_format_legacy or malware name as a prefix is acceptable if wanting to feature
separate from others. Noting this may limit correlation across families. Ask yourself, if
there was another plugin or malware family with the same metadata, would you want to correlate
with it?

- eg. `pe_compile_time` not `lief_pe_compile_time`

- eg. `signature_signer_thumbprint` not `sigcheck_signer_thumbprint` nor `pe_signer_thumbprint`
  (as same signer could be used on macros, macho, etc.)

- eg. `obfuscation_scheme` not `malcarve_obfuscation`

- eg. `config_callback_domain` preferred over `zeus_c2`

### Meaningful, specific feature names over key-value style

This one is not always possible or practical but refers to having feature names that are so
generic that all values output from a tool are capture by one/two features.

Note: Azul _do_ have features like this for some generic plugins such as `exiftool` and `tika` but
these are generic file metadata extractors where the alternative of predefining every metadata
key as a feature is not feasible.

Sometimes plugins need to do a combination of both styles to capture all available metadata
from a tool/decoder but also map specific values to well-known feature names.

- eg. `config_callback_domain` not `notepad_config_value`: `label` = _domain_

- eg. `config_persistence_key` not `spyeye_string`

### Boolean values are poor choices as feature values

Features should not be defined/named as flags. Instead, if required, there is a generic
feature named `tag` that can be set with string value of the information that is being flagged.

- eg. `tag` = _antivm_checks_ not `has_antivm_checks` = _True_

- eg. `config_disabled_feature` = _display_msgbox_ not `config_display_msgbox` = _False_

### Features should derive from content, not be inferred by the author

Authors should resist interpreting/labelling information that is not derivable from a sample.

This can be a grey area but an example would be, a feature shouldn't be for a named APT
group but could be after a malware family. i.e. Nothing from the sample is telling you who used
the malware and tools can be shared between actors. Likewise an exploitation technique that
is currently used by a single actor will likely be adopted more widely in the future.

Linkages that can change over time usually make poor features.

## Cross-plugin Correlation

In Azul 3 developers are aiming to improve correlation of samples where different tools may extract the
same logical metadata. In particular, with the ability to bulk import/feature metadata from external
sources like VirusTotal this opens up the ability to correlate a locally processed binary with
something Azul only has metadata for.

To this end, plugin authors should be striving to match with existing features and feature value
normalisation for metadata that already exists in Azul.

Look in the Azul UI at Features -> Explore to see what is already defined.

Config decoders are a prime use case where feature reuse allows matching values across malware
families. eg. reused service names, obfuscation/encryption keys, network infrastructure, etc.

## Handling Structured Metadata

In the real world, metadata is not always nice flat key-value pairs. An author may need to think
of ways to flatten structured information without losing the detail of what is being featured.

Feature values can contain an additional string _label_, which can usually help in these situations.

For example if an author needs to feature different certificate hash schemes and a sample can contain
more than one certificate, it may be necessary or helpful to label with the certificate thumbprint
or other field to distinguish which one it belongs to.

Likewise a document may have been created by a specific application and version. These could be raised
as separate features with the version feature labeled by the application name.

Unlike Azul 2, labels do not need to (but can) exist as their own defined features and now must
be string types.

What if the metadata is heavily structured and simplifying to key:value pairs loses too much detail?

Plugins can output a json formatted `info` object from processing (See _Runner Usage_) as well.
The author should map what they can to features that would help with analysis/correlation and keep
the full structure in `info`. It will not help in correlation and is not indexed but could be used
in custom UI widgets to display to analysts.

## Feature Types

There are a number of primitive data types for features and a couple of higher-level types to
allow correlation across features.

- string (UTF-8 encodable string)
- integer (signed 64bit)
- float (double-precision 64bit IEEE 754)
- bytes (indexed as base64 and will attempt to render as escaped string in UI)
- datetime (timestamp in ISO8601 format)
- filepath (UTF-8 encodable absolute or relative filepath, will be decomposed into subparts for indexing)
- uri (UTF-8 encodable network URL/domain/hostname, will be decomposed into subparts for indexing)

Note: The plugin client/library you are using may provide language specific type mappings for these.
Refer to _Runner Usage_ documentation.

The **filepath** and **uri** types are decomposed at index time and the UI provides additional
functionality to correlate with their values/subparts across named features. This can be useful
to pivot to related samples that do not necessarily share the exact same feature and value.

eg. pivot from a malware sample's callback domain to samples that were sourced from the same host,
or finding related samples based on a shared pdb debug path prefix.

## Conclusion

With a little forethought plugin authors can define new features in a consistent and complimentary
way to existing plugins.

At the end of the day the authors are best placed to answer what data they want displayed via the
UI and how they want to be able to pivot and correlate with other entities in the system.

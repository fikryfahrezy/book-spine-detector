# Cross-platform design contract

`tokens.json` is the canonical visual contract for the Android Compose app and the future SwiftUI
app. Values are semantic rather than framework-specific: `surface`, `textPrimary`, `accent`, and
status colors describe intent, not Material or UIKit roles.

Android maps the contract in `ui/theme/Tokens.kt`. A future iOS target should create an equivalent
`AppTokens` type with `Color`, `CGFloat`, and `Duration` values while preserving the names and raw
values. The app deliberately avoids dynamic system colors so factory devices render the same scan
states on both platforms.

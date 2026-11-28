import AppKit
import Carbon

@MainActor
final class Hotkeys {
    private static weak var sharedInstance: Hotkeys?
    private var hotKeyRefs: [EventHotKeyRef?] = []
    private var handlerRef: EventHandlerRef?
    private var onToggleIsland: (@MainActor () -> Void)?
    private var onOpenWebGui: (@MainActor () -> Void)?

    func register(
        onToggleIsland: @escaping @MainActor () -> Void,
        onOpenWebGui: @escaping @MainActor () -> Void
    ) {
        self.onToggleIsland = onToggleIsland
        self.onOpenWebGui = onOpenWebGui
        Self.sharedInstance = self

        var eventType = EventTypeSpec(
            eventClass: OSType(kEventClassKeyboard),
            eventKind: UInt32(kEventHotKeyPressed)
        )

        var ref: EventHandlerRef?
        let status = InstallEventHandler(
            GetApplicationEventTarget(),
            { _, event, _ -> OSStatus in
                var hotKeyID = EventHotKeyID()
                let err = GetEventParameter(
                    event,
                    EventParamName(kEventParamDirectObject),
                    EventParamType(typeEventHotKeyID),
                    nil,
                    MemoryLayout<EventHotKeyID>.size,
                    nil,
                    &hotKeyID
                )
                guard err == noErr else { return err }
                let id = hotKeyID.id
                Task { @MainActor in
                    guard let instance = Hotkeys.sharedInstance else { return }
                    if id == 1 {
                        instance.onToggleIsland?()
                    } else if id == 2 {
                        instance.onOpenWebGui?()
                    }
                }
                return noErr
            },
            1,
            &eventType,
            nil,
            &ref
        )
        if status == noErr {
            handlerRef = ref
        }

        let mods = UInt32(controlKey | optionKey)
        registerSingle(id: 1, keyCode: UInt32(kVK_ANSI_M), modifiers: mods)
        registerSingle(id: 2, keyCode: UInt32(kVK_ANSI_G), modifiers: mods)
    }

    private func registerSingle(id: UInt32, keyCode: UInt32, modifiers: UInt32) {
        let hotKeyID = EventHotKeyID(signature: 0x474D4B49, id: id) // 'GMKI'
        var ref: EventHotKeyRef?
        let err = RegisterEventHotKey(keyCode, modifiers, hotKeyID, GetApplicationEventTarget(), 0, &ref)
        if err == noErr {
            hotKeyRefs.append(ref)
        }
    }

    deinit {
        for ref in hotKeyRefs {
            if let ref { UnregisterEventHotKey(ref) }
        }
        if let handlerRef {
            RemoveEventHandler(handlerRef)
        }
    }
}

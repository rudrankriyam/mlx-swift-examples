//
//  MessageView.swift
//  MLXChatExample
//
//  Created by İbrahim Çetin on 20.04.2025.
//

import AVKit
import SwiftUI

/// A view that displays a single message in the chat interface.
/// Supports different message roles (user, assistant, system) and media attachments.
struct MessageView: View {
    /// The message to be displayed
    let message: Message

    /// Creates a message view
    /// - Parameter message: The message model to display
    init(_ message: Message) {
        self.message = message
    }

    private func splitThinking(_ content: String) -> (thinking: String?, after: String?) {
        guard let startRange = content.range(of: "<think>") else {
            return (nil, content.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        guard let endRange = content.range(of: "</think>") else {
            let thinking = String(content[startRange.upperBound...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return (thinking, nil)
        }

        let thinking = String(content[startRange.upperBound ..< endRange.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let afterThink = String(content[endRange.upperBound...])
            .trimmingCharacters(in: .whitespacesAndNewlines)

        return (thinking, afterThink.isEmpty ? nil : afterThink)
    }

    var body: some View {
        switch message.role {
        case .user:
            // User messages are right-aligned with blue background
            HStack {
                Spacer()
                VStack(alignment: .trailing, spacing: 8) {
                    // Display first image if present
                    if let firstImage = message.images.first {
                        AsyncImage(url: firstImage) { image in
                            image
                                .resizable()
                                .aspectRatio(contentMode: .fill)
                        } placeholder: {
                            ProgressView()
                        }
                        .frame(maxWidth: 250, maxHeight: 200)
                        .clipShape(.rect(cornerRadius: 12))
                    }

                    // Display first video if present
                    if let firstVideo = message.videos.first {
                        VideoPlayer(player: AVPlayer(url: firstVideo))
                            .frame(width: 250, height: 340)
                            .clipShape(.rect(cornerRadius: 12))
                    }

                    // Message content with tinted background.
                    // LocalizedStringKey used to trigger default handling of markdown content.
                    Text(LocalizedStringKey(message.content))
                        .padding(.vertical, 8)
                        .padding(.horizontal, 12)
                        .background(.tint, in: .rect(cornerRadius: 16))
                        .textSelection(.enabled)
                }
            }

        case .assistant:
            // Assistant messages are left-aligned without background
            // LocalizedStringKey used to trigger default handling of markdown content.
            HStack {
                let (thinking, afterThink) = splitThinking(message.content)
                VStack(alignment: .leading, spacing: 8) {
                    if let thinking {
                        Text(LocalizedStringKey(thinking))
                            .italic()
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }

                    if let afterThink {
                        Text(LocalizedStringKey(afterThink))
                            .textSelection(.enabled)
                    } else if thinking == nil {
                        Text(LocalizedStringKey(message.content))
                            .textSelection(.enabled)
                    }
                }

                Spacer()
            }

        case .system:
            // System messages are centered with computer icon
            Label(message.content, systemImage: "desktopcomputer")
                .font(.headline)
                .foregroundColor(.secondary)
                .frame(maxWidth: .infinity, alignment: .center)
        }
    }
}

#Preview {
    VStack(spacing: 20) {
        MessageView(.system("You are a helpful assistant."))

        MessageView(
            .user(
                "Here's a photo",
                images: [URL(string: "https://picsum.photos/200")!]
            )
        )

        MessageView(.assistant("I see your photo!"))
    }
    .padding()
}

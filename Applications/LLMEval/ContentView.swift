// Copyright © 2024 Apple Inc.

import SwiftUI

struct ContentView: View {
    @StateObject private var streamingViewModel = ParallelStreamingViewModel()

    var body: some View {
        ParallelStreamingComparisonView(viewModel: streamingViewModel)
    }
}

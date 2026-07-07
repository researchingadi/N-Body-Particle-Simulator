import { useEffect } from 'react'
import { SimulationCanvas } from './components/SimulationCanvas'
import { TopBar } from './components/TopBar'
import { ControlPanel } from './components/ControlPanel'
import { DiagnosticsPanel } from './components/DiagnosticsPanel'
import { Legend } from './components/Legend'
import { useSimulationStore } from './store/simulationStore'

function App() {
  const loadCatalog = useSimulationStore((s) => s.loadCatalog)

  useEffect(() => {
    void loadCatalog()
  }, [loadCatalog])

  return (
    <div className="relative h-screen w-screen">
      {/* Full-bleed 3D stage -- the canvas IS the content, not a boxed widget. */}
      <div className="absolute inset-0">
        <SimulationCanvas />
      </div>

      {/* Floating HUD overlay, laid out like a mission-control interface. */}
      <div className="absolute inset-0 pointer-events-none flex flex-col justify-between p-4">
        <div className="flex justify-between items-start gap-4">
          <TopBar />
          <DiagnosticsPanel />
        </div>
        <div className="flex justify-between items-end gap-4">
          <ControlPanel />
          <Legend />
        </div>
      </div>
    </div>
  )
}

export default App

import {
  cropAndResizeImageTensor,
  getImageFeatures,
  loadImageModel,
  tf,
} from 'tensorflow-helpers/dist/browser'

// Interfaces
interface ShapeProperties {
  type: ShapeType | null
  x: number
  y: number
  size: number
  width: number
  height: number
  rotation: number
  color: string
}

interface Point {
  x: number
  y: number
}

type ShapeType = 'square' | 'rectangle' | 'triangle' | 'circle'
type ExportFormat = 'png' | 'jpeg'

// Status types
type StatusType = 'ready' | 'loading' | 'error' | 'success'

class GraphEditor {
  private originalCanvas: HTMLCanvasElement
  private editorCanvas: HTMLCanvasElement
  private originalCtx: CanvasRenderingContext2D
  private editorCtx: CanvasRenderingContext2D
  private backgroundImage: HTMLImageElement | null = null
  private currentShape: ShapeType | null = null
  private isDragging = false
  private isResizing = false
  private isRotating = false
  private dragOffset: Point = { x: 0, y: 0 }
  private mousePosition: Point = { x: 0, y: 0 }
  private resizeHandle: string | null = null // 'nw', 'ne', 'sw', 'se', 'n', 'e', 's', 'w'
  private lockRatio = true

  private shapeProperties: ShapeProperties = {
    type: null,
    x: 300,
    y: 210,
    size: 1,
    width: 120,
    height: 80,
    rotation: 0,
    color: '#ffffff',
  }

  // Grid display properties
  private gridProperties = {
    lineColor: '#3b82f6',
    textColor: '#ffffff',
    confColor: '#ff0000ff',
    textSize: 10,
  }

  private imageModel: any

  // Spatial data in 7*7*160
  private spatialData: number[][][] = Array.from({ length: 7 }, () =>
    Array.from({ length: 7 }, () => Array.from({ length: 160 }, () => 0)),
  )

  //grid = 7*7
  private gridData: number[][] = Array.from({ length: 7 }, () =>
    Array.from({ length: 7 }, () => 0),
  )

  private gridStat = { min: 0, max: 0, mean: 0, stdDev: 0 }

  private gridDataType = {
    type: 'feat' as 'feat' | 'Euclidean' | 'cos',
    feat_num: 0,
  }

  // DOM Elements
  private elements: { [key: string]: HTMLElement } = {}

  constructor() {
    this.originalCanvas = document.getElementById(
      'original-canvas',
    ) as HTMLCanvasElement
    this.editorCanvas = document.getElementById(
      'editor-canvas',
    ) as HTMLCanvasElement
    this.originalCtx = this.originalCanvas.getContext('2d')!
    this.editorCtx = this.editorCanvas.getContext('2d')!

    // Set editor canvas to have a bright white background like the original
    this.editorCanvas.style.backgroundColor = '#ffffff'

    this.initializeElements()
    this.initializeModel()
    this.setupEventListeners()
    this.updateStatus('ready', 'Ready - Import an image to get started')

    console.log('Graph Editor initialized')
  }

  private initializeElements(): void {
    // Cache frequently used DOM elements
    const elementIds = [
      'file-upload',
      'url-input',
      'url-load-btn',
      'predefined-select',
      'color-picker',
      'clear-shape-btn',
      'save-btn',
      'original-canvas-size',
      'grid-size',
      'mouse-coords',
      'zoom-level',
      'status-text',
      'status-indicator',
      'original-canvas-container',
      'editor-canvas-container',
      'drop-zone',
      'canvas-overlay',
      'grid-line-color',
      'grid-text-color',
      'grid-text-size',
      'grid-text-size-value',
      'size-input-bottom',
      'width-input-bottom',
      'height-input-bottom',
      'rotation-input-bottom',
      'width-control-bottom',
      'height-control-bottom',
      'rotation-control-bottom',
      'lock-ratio-checkbox',
      'feat-num-input',
    ]

    elementIds.forEach(id => {
      const element = document.getElementById(id)
      if (element) {
        this.elements[id] = element
      }
    })
  }

  private async initializeModel() {
    this.imageModel = await loadImageModel({
      url: '/saved_models/base_model',
      cacheUrl: 'indexeddb://base_model',
    })
  }

  private setupEventListeners(): void {
    // File upload
    this.elements['file-upload']?.addEventListener('change', e => {
      const input = e.target as HTMLInputElement
      if (input.files && input.files[0]) {
        this.handleFileUpload(input.files[0])
      }
      this.updateGridData()
    })

    // URL loading
    this.elements['url-load-btn']?.addEventListener('click', () => {
      const urlInput = this.elements['url-input'] as HTMLInputElement
      const url = urlInput.value.trim()
      if (url) {
        this.loadImageFromURL(url)
      }
      this.updateGridData()
    })

    // Enter key for URL input
    this.elements['url-input']?.addEventListener('keypress', e => {
      if (e.key === 'Enter') {
        this.elements['url-load-btn']?.click()
      }
      this.updateGridData()
    })

    // Shape selection buttons
    document.querySelectorAll('.shape-btn').forEach(btn => {
      btn.addEventListener('click', e => {
        const button = e.currentTarget as HTMLButtonElement
        const shapeType = button.dataset.shape as ShapeType
        this.selectShape(shapeType)
        this.updateGridData()
      })
    })

    // Control inputs
    this.setupControlListeners()

    // Canvas events
    this.setupCanvasEvents()

    // Color controls
    this.setupColorControls()

    // Action buttons
    this.elements['clear-shape-btn']?.addEventListener('click', () => {
      this.clearShape()
      this.updateGridData()
    })

    this.elements['save-btn']?.addEventListener('click', () => {
      this.saveCanvas()
    })

    // Drag and drop
    this.setupDragAndDrop()

    // Grid color controls
    this.setupGridColorControls()

    // Window resize
    window.addEventListener('resize', () => {
      this.handleWindowResize()
      this.updateGridData()
    })
  }

  private setupControlListeners(): void {
    // Bottom control inputs (text inputs only)
    this.elements['size-input-bottom']?.addEventListener('input', e => {
      const value = parseFloat((e.target as HTMLInputElement).value)
      this.updateShapeProperty('size', value)
      this.updateGridData()
    })

    this.elements['width-input-bottom']?.addEventListener('input', e => {
      const value = parseFloat((e.target as HTMLInputElement).value)
      this.updateShapeProperty('width', value)
      this.updateGridData()
    })

    this.elements['height-input-bottom']?.addEventListener('input', e => {
      const value = parseFloat((e.target as HTMLInputElement).value)
      this.updateShapeProperty('height', value)
      this.updateGridData()
    })

    this.elements['rotation-input-bottom']?.addEventListener('input', e => {
      const value = parseFloat((e.target as HTMLInputElement).value)
      this.updateShapeProperty('rotation', value)
      this.updateGridData()
    })

    // Lock ratio checkbox
    this.elements['lock-ratio-checkbox']?.addEventListener('change', e => {
      this.lockRatio = (e.target as HTMLInputElement).checked
    })

    this.elements['feat-num-input']?.addEventListener('input', e => {
      const value = parseFloat((e.target as HTMLInputElement).value)
      this.gridDataType.feat_num = value < 0 ? 0 : value > 159 ? 159 : value
      this.updateGridData()
    })
  }

  private setupCanvasEvents(): void {
    // Mouse events for shape dragging on editor canvas
    this.editorCanvas.addEventListener('mousedown', e => {
      this.handleMouseDown(e)
      this.updateGridData()
    })
    this.editorCanvas.addEventListener('mousemove', e => {
      this.handleMouseMove(e)
      this.updateGridData()
    })
    this.editorCanvas.addEventListener('mouseup', () => {
      this.handleMouseUp()
      this.updateGridData()
    })
    this.editorCanvas.addEventListener('mouseleave', () => {
      this.handleMouseLeave()
      this.updateGridData()
    })

    // Touch events for mobile support
    this.editorCanvas.addEventListener('touchstart', e => {
      this.handleTouchStart(e)
      this.updateGridData()
    })
    this.editorCanvas.addEventListener('touchmove', e => {
      this.handleTouchMove(e)
      this.updateGridData()
    })
    this.editorCanvas.addEventListener('touchend', () => {
      this.handleTouchEnd()
      this.updateGridData()
    })
  }

  private setupColorControls(): void {
    // Color picker
    this.elements['color-picker']?.addEventListener('change', e => {
      const input = e.target as HTMLInputElement
      this.updateShapeProperty('color', input.value)
      this.updateGridData()
    })

    // Color presets
    document.querySelectorAll('.color-preset').forEach(btn => {
      btn.addEventListener('click', e => {
        const button = e.target as HTMLButtonElement
        const color = button.dataset.color
        if (color) {
          this.updateShapeProperty('color', color)
          ;(this.elements['color-picker'] as HTMLInputElement).value = color
          this.updateGridData()
        }
      })
    })
  }

  private setupGridColorControls(): void {
    // Grid line color picker
    this.elements['grid-line-color']?.addEventListener('change', e => {
      const input = e.target as HTMLInputElement
      this.gridProperties.lineColor = input.value
      this.redrawGridClassification()
    })

    // Grid text color picker
    this.elements['grid-text-color']?.addEventListener('change', e => {
      const input = e.target as HTMLInputElement
      this.gridProperties.textColor = input.value
      this.redrawGridClassification()
    })

    // Grid text size slider
    this.elements['grid-text-size']?.addEventListener('input', e => {
      const input = e.target as HTMLInputElement
      this.gridProperties.textSize = parseInt(input.value)
      const valueDisplay = this.elements['grid-text-size-value']
      if (valueDisplay) {
        valueDisplay.textContent = `${input.value}px`
      }
      this.redrawGridClassification()
    })
  }

  private setupDragAndDrop(): void {
    const canvasContainer = this.elements['original-canvas-container']
    const dropZone = this.elements['drop-zone']
    const overlay = this.elements['canvas-overlay']

    if (!canvasContainer || !dropZone || !overlay)
      return // Prevent default drag behaviors
    ;['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      canvasContainer.addEventListener(eventName, e => e.preventDefault())
      document.body.addEventListener(eventName, e => e.preventDefault())
    })

    // Handle drag enter/over
    canvasContainer.addEventListener('dragenter', () => {
      overlay.classList.add('active')
    })

    canvasContainer.addEventListener('dragover', () => {
      overlay.classList.add('active')
    })

    // Handle drag leave
    canvasContainer.addEventListener('dragleave', e => {
      if (!canvasContainer.contains(e.relatedTarget as Node)) {
        overlay.classList.remove('active')
      }
    })

    // Handle drop
    canvasContainer.addEventListener('drop', e => {
      overlay.classList.remove('active')
      const files = e.dataTransfer?.files
      if (files && files.length > 0 && files[0].type.indexOf('image/') === 0) {
        this.handleFileUpload(files[0])
      }
    })
  }

  private getResizeHandle(x: number, y: number): string | null {
    if (!this.currentShape) return null

    const props = this.shapeProperties
    const handleSize = 8

    let left, right, top, bottom

    switch (props.type) {
      case 'square':
        const squareSize = 40 * props.size
        left = props.x - squareSize
        right = props.x + squareSize
        top = props.y - squareSize
        bottom = props.y + squareSize
        break
      case 'rectangle':
        left = props.x - props.width / 2
        right = props.x + props.width / 2
        top = props.y - props.height / 2
        bottom = props.y + props.height / 2
        break
      case 'circle':
        const radius = 40 * props.size
        left = props.x - radius
        right = props.x + radius
        top = props.y - radius
        bottom = props.y + radius
        break
      case 'triangle':
        const triangleSize = 40 * props.size
        left = props.x - triangleSize
        right = props.x + triangleSize
        top = props.y - triangleSize
        bottom = props.y + triangleSize
        break
      default:
        return null
    }

    // Check corner handles
    if (Math.abs(x - left) <= handleSize && Math.abs(y - top) <= handleSize)
      return 'nw'
    if (Math.abs(x - right) <= handleSize && Math.abs(y - top) <= handleSize)
      return 'ne'
    if (Math.abs(x - left) <= handleSize && Math.abs(y - bottom) <= handleSize)
      return 'sw'
    if (Math.abs(x - right) <= handleSize && Math.abs(y - bottom) <= handleSize)
      return 'se'

    // Check edge handles (only for rectangle)
    if (props.type === 'rectangle') {
      if (
        Math.abs(x - props.x) <= handleSize &&
        Math.abs(y - top) <= handleSize
      )
        return 'n'
      if (
        Math.abs(x - right) <= handleSize &&
        Math.abs(y - props.y) <= handleSize
      )
        return 'e'
      if (
        Math.abs(x - props.x) <= handleSize &&
        Math.abs(y - bottom) <= handleSize
      )
        return 's'
      if (
        Math.abs(x - left) <= handleSize &&
        Math.abs(y - props.y) <= handleSize
      )
        return 'w'
    }

    return null
  }

  private getResizeCursor(handle: string): string {
    switch (handle) {
      case 'nw':
      case 'se':
        return 'nw-resize'
      case 'ne':
      case 'sw':
        return 'ne-resize'
      case 'n':
      case 's':
        return 'ns-resize'
      case 'e':
      case 'w':
        return 'ew-resize'
      default:
        return 'default'
    }
  }

  private isPointInRotationHandle(x: number, y: number): boolean {
    if (!this.currentShape || this.shapeProperties.type === 'circle')
      return false

    const props = this.shapeProperties
    const handleSize = 8
    let top

    switch (props.type) {
      case 'square':
        const squareSize = 40 * props.size
        top = props.y - squareSize - 20 // 20px above the shape
        break
      case 'rectangle':
        top = props.y - props.height / 2 - 20
        break
      case 'triangle':
        const triangleSize = 40 * props.size
        top = props.y - triangleSize - 20
        break
      default:
        return false
    }

    return (
      Math.abs(x - props.x) <= handleSize && Math.abs(y - top) <= handleSize
    )
  }

  private handleResize(mouseX: number, mouseY: number): void {
    if (!this.resizeHandle) return

    const props = this.shapeProperties
    const deltaX = mouseX - this.dragOffset.x
    const deltaY = mouseY - this.dragOffset.y

    switch (props.type) {
      case 'square':
      case 'circle':
        // For square and circle, resize uniformly based on handle direction
        let sizeDelta = 0
        const moveDistance = Math.max(Math.abs(deltaX), Math.abs(deltaY))

        // Determine direction based on resize handle and movement
        if (this.resizeHandle === 'se') {
          // Southeast: dragging away from center = grow, towards center = shrink
          sizeDelta =
            deltaX > 0 || deltaY > 0
              ? moveDistance * 0.01
              : -moveDistance * 0.01
        } else if (this.resizeHandle === 'nw') {
          // Northwest: dragging away from center = grow, towards center = shrink
          sizeDelta =
            deltaX < 0 || deltaY < 0
              ? moveDistance * 0.01
              : -moveDistance * 0.01
        } else if (this.resizeHandle === 'ne') {
          // Northeast: dragging away from center = grow, towards center = shrink
          sizeDelta =
            deltaX > 0 || deltaY < 0
              ? moveDistance * 0.01
              : -moveDistance * 0.01
        } else if (this.resizeHandle === 'sw') {
          // Southwest: dragging away from center = grow, towards center = shrink
          sizeDelta =
            deltaX < 0 || deltaY > 0
              ? moveDistance * 0.01
              : -moveDistance * 0.01
        }

        const newSize = Math.max(0.2, props.size + sizeDelta)
        this.updateShapeProperty('size', newSize)
        break

      case 'rectangle':
        let newWidth = props.width
        let newHeight = props.height

        if (this.resizeHandle.indexOf('e') !== -1) newWidth += deltaX
        if (this.resizeHandle.indexOf('w') !== -1) newWidth -= deltaX
        if (this.resizeHandle.indexOf('n') !== -1) newHeight -= deltaY
        if (this.resizeHandle.indexOf('s') !== -1) newHeight += deltaY

        if (this.lockRatio) {
          const ratio = props.width / props.height
          if (Math.abs(deltaX) > Math.abs(deltaY)) {
            newHeight = newWidth / ratio
          } else {
            newWidth = newHeight * ratio
          }
        }

        this.updateShapeProperty('width', Math.max(20, newWidth))
        this.updateShapeProperty('height', Math.max(20, newHeight))
        break

      case 'triangle':
        // For triangle, resize uniformly based on handle direction
        let triangleSizeDelta = 0
        const triangleMoveDistance = Math.max(
          Math.abs(deltaX),
          Math.abs(deltaY),
        )

        // Determine direction based on resize handle and movement
        if (this.resizeHandle === 'se') {
          // Southeast: dragging away from center = grow, towards center = shrink
          triangleSizeDelta =
            deltaX > 0 || deltaY > 0
              ? triangleMoveDistance * 0.01
              : -triangleMoveDistance * 0.01
        } else if (this.resizeHandle === 'nw') {
          // Northwest: dragging away from center = grow, towards center = shrink
          triangleSizeDelta =
            deltaX < 0 || deltaY < 0
              ? triangleMoveDistance * 0.01
              : -triangleMoveDistance * 0.01
        } else if (this.resizeHandle === 'ne') {
          // Northeast: dragging away from center = grow, towards center = shrink
          triangleSizeDelta =
            deltaX > 0 || deltaY < 0
              ? triangleMoveDistance * 0.01
              : -triangleMoveDistance * 0.01
        } else if (this.resizeHandle === 'sw') {
          // Southwest: dragging away from center = grow, towards center = shrink
          triangleSizeDelta =
            deltaX < 0 || deltaY > 0
              ? triangleMoveDistance * 0.01
              : -triangleMoveDistance * 0.01
        }

        const newTriangleSize = Math.max(0.2, props.size + triangleSizeDelta)
        this.updateShapeProperty('size', newTriangleSize)
        break
    }

    this.dragOffset.x = mouseX
    this.dragOffset.y = mouseY
    this.updateControlValues()
  }

  private handleRotation(mouseX: number, mouseY: number): void {
    const props = this.shapeProperties
    const centerX = props.x
    const centerY = props.y

    const angle1 = Math.atan2(
      this.dragOffset.y - centerY,
      this.dragOffset.x - centerX,
    )
    const angle2 = Math.atan2(mouseY - centerY, mouseX - centerX)
    const deltaAngle = (angle2 - angle1) * (180 / Math.PI)

    const newRotation = (props.rotation + deltaAngle) % 360
    this.updateShapeProperty(
      'rotation',
      newRotation < 0 ? newRotation + 360 : newRotation,
    )

    this.dragOffset.x = mouseX
    this.dragOffset.y = mouseY
    this.updateControlValues()
  }

  private handleMouseDown(e: MouseEvent): void {
    if (!this.currentShape) return

    const rect = this.editorCanvas.getBoundingClientRect()
    const mouseX = e.clientX - rect.left
    const mouseY = e.clientY - rect.top

    // Check for resize handles first
    const resizeHandle = this.getResizeHandle(mouseX, mouseY)
    if (resizeHandle) {
      this.isResizing = true
      this.resizeHandle = resizeHandle
      this.dragOffset.x = mouseX
      this.dragOffset.y = mouseY
      this.editorCanvas.style.cursor = this.getResizeCursor(resizeHandle)
      return
    }

    // Check for rotation handle
    if (this.isPointInRotationHandle(mouseX, mouseY)) {
      this.isRotating = true
      this.dragOffset.x = mouseX
      this.dragOffset.y = mouseY
      this.editorCanvas.style.cursor = 'grab'
      return
    }

    // Check for shape dragging
    if (this.isPointInShape(mouseX, mouseY)) {
      this.isDragging = true
      this.dragOffset.x = mouseX - this.shapeProperties.x
      this.dragOffset.y = mouseY - this.shapeProperties.y
      this.editorCanvas.style.cursor = 'grabbing'
      this.editorCanvas.classList.add('dragging')
    }
  }

  private handleMouseMove(e: MouseEvent): void {
    const rect = this.editorCanvas.getBoundingClientRect()
    const mouseX = e.clientX - rect.left
    const mouseY = e.clientY - rect.top

    // Update mouse coordinates display
    this.mousePosition = { x: Math.round(mouseX), y: Math.round(mouseY) }
    this.updateMouseCoordinates()

    if (this.isResizing && this.resizeHandle) {
      this.handleResize(mouseX, mouseY)
    } else if (this.isRotating) {
      this.handleRotation(mouseX, mouseY)
    } else if (this.isDragging) {
      // Update shape position
      this.shapeProperties.x = mouseX - this.dragOffset.x
      this.shapeProperties.y = mouseY - this.dragOffset.y
      this.redraw()
      this.updateControlValues()
    } else if (this.currentShape) {
      // Update cursor based on hover state
      const resizeHandle = this.getResizeHandle(mouseX, mouseY)
      if (resizeHandle) {
        this.editorCanvas.style.cursor = this.getResizeCursor(resizeHandle)
      } else if (this.isPointInRotationHandle(mouseX, mouseY)) {
        this.editorCanvas.style.cursor = 'grab'
      } else if (this.isPointInShape(mouseX, mouseY)) {
        this.editorCanvas.style.cursor = 'grab'
      } else {
        this.editorCanvas.style.cursor = 'crosshair'
      }
    }
  }

  private handleMouseUp(): void {
    if (this.isDragging || this.isResizing || this.isRotating) {
      this.isDragging = false
      this.isResizing = false
      this.isRotating = false
      this.resizeHandle = null
      this.editorCanvas.style.cursor = this.currentShape
        ? 'crosshair'
        : 'crosshair'
      this.editorCanvas.classList.remove('dragging')
    }
  }

  private handleMouseLeave(): void {
    this.handleMouseUp()
  }

  // Touch event handlers for mobile support
  private handleTouchStart(e: TouchEvent): void {
    e.preventDefault()
    const touch = e.touches[0]
    this.handleMouseDown({
      clientX: touch.clientX,
      clientY: touch.clientY,
    } as MouseEvent)
  }

  private handleTouchMove(e: TouchEvent): void {
    e.preventDefault()
    const touch = e.touches[0]
    this.handleMouseMove({
      clientX: touch.clientX,
      clientY: touch.clientY,
    } as MouseEvent)
  }

  private handleTouchEnd(): void {
    this.handleMouseUp()
  }

  private handleWindowResize(): void {
    // Handle window resize if needed
    this.updateCanvasInfo()
  }

  private handleFileUpload(file: File): void {
    if (file.type.indexOf('image/') !== 0) {
      this.updateStatus('error', 'Please select a valid image file')
      return
    }

    this.updateStatus('loading', 'Loading image...')

    const reader = new FileReader()
    reader.onload = e => {
      if (e.target?.result) {
        this.loadImageFromDataURL(e.target.result as string)
      }
    }
    reader.onerror = () => {
      this.updateStatus('error', 'Failed to read file')
    }
    reader.readAsDataURL(file)
  }

  private loadImageFromURL(url: string): void {
    this.updateStatus('loading', 'Loading image from URL...')

    const img = new Image()
    img.onload = () => {
      this.setBackgroundImage(img)
    }
    img.onerror = () => {
      this.updateStatus('error', 'Failed to load image from URL')
    }
    img.crossOrigin = 'anonymous'
    img.src = url
  }

  private loadImageFromDataURL(dataURL: string): void {
    const img = new Image()
    img.onload = () => {
      this.setBackgroundImage(img)
    }
    img.onerror = () => {
      this.updateStatus('error', 'Failed to load image')
    }
    img.src = dataURL
  }

  private setBackgroundImage(img: HTMLImageElement): void {
    this.backgroundImage = img

    // Set both canvases to the same size
    const maxWidth = 600
    const maxHeight = 420
    const ratio = Math.min(maxWidth / img.width, maxHeight / img.height)

    const canvasWidth = Math.round(img.width * ratio)
    const canvasHeight = Math.round(img.height * ratio)

    // Update both canvases
    this.originalCanvas.width = canvasWidth
    this.originalCanvas.height = canvasHeight
    this.editorCanvas.width = canvasWidth
    this.editorCanvas.height = canvasHeight

    // Draw the 7x7 grid classification on the original canvas
    this.drawGridClassification(img, canvasWidth, canvasHeight)

    // Reset shape position to center of editor canvas
    this.shapeProperties.x = canvasWidth / 2
    this.shapeProperties.y = canvasHeight / 2

    this.updateCanvasInfo()
    this.redraw()
    this.updateStatus('success', 'Image loaded and classified into 7×7 grid')
  }

  private drawGridClassification(
    img: HTMLImageElement,
    canvasWidth: number,
    canvasHeight: number,
  ): void {
    // Clear the original canvas
    this.originalCtx.clearRect(0, 0, canvasWidth, canvasHeight)

    // Calculate grid dimensions
    const gridCols = 7
    const gridRows = 7
    const cellWidth = canvasWidth / gridCols
    const cellHeight = canvasHeight / gridRows

    // Calculate source image dimensions per grid cell
    const srcCellWidth = img.width / gridCols
    const srcCellHeight = img.height / gridRows

    const startR = parseInt(this.gridProperties.confColor.slice(1, 3), 16)
    const startG = parseInt(this.gridProperties.confColor.slice(3, 5), 16)
    const startB = parseInt(this.gridProperties.confColor.slice(5, 7), 16)
    const startA = parseInt(this.gridProperties.confColor.slice(7, 9), 16)

    // Draw each grid cell
    for (let row = 0; row < gridRows; row++) {
      for (let col = 0; col < gridCols; col++) {
        // Source coordinates in the original image
        const srcX = col * srcCellWidth
        const srcY = row * srcCellHeight

        // Destination coordinates on the canvas
        const destX = col * cellWidth
        const destY = row * cellHeight

        // Draw the image portion for this grid cell
        this.originalCtx.drawImage(
          img,
          srcX,
          srcY,
          srcCellWidth,
          srcCellHeight, // Source rectangle
          destX,
          destY,
          cellWidth,
          cellHeight, // Destination rectangle
        )

        // Draw dotted grid lines
        this.originalCtx.strokeStyle = this.gridProperties.lineColor
        this.originalCtx.lineWidth = 1
        this.originalCtx.setLineDash([3, 3]) // Create dotted line pattern
        this.originalCtx.strokeRect(destX, destY, cellWidth, cellHeight)

        // Add grid cell labels with confusion matrix values (placeholder using deterministic random)
        const confusionValue = this.gridData[row][col]
        const displayValue = confusionValue.toFixed(3) // Show 3 decimal places

        this.originalCtx.font = `${this.gridProperties.textSize}px Arial`

        // Interpolate each component linearly
        const r = Math.round(
          startR *
            ((confusionValue - this.gridStat.min) /
              (this.gridStat.max - this.gridStat.min)),
        )
        const g = Math.round(
          startG *
            ((confusionValue - this.gridStat.min) /
              (this.gridStat.max - this.gridStat.min)),
        )
        const b = Math.round(
          startB *
            ((confusionValue - this.gridStat.min) /
              (this.gridStat.max - this.gridStat.min)),
        )
        const a = Math.round(
          startA *
            ((confusionValue - this.gridStat.min) /
              (this.gridStat.max - this.gridStat.min)),
        )
        const gradient = `#${r.toString(16).padStart(2, '0')}${g
          .toString(16)
          .padStart(2, '0')}${b.toString(16).padStart(2, '0')}${a
          .toString(16)
          .padStart(2, '0')}`
        this.originalCtx.fillStyle = gradient
        this.originalCtx.fillRect(destX, destY, cellWidth, cellHeight)

        // Calculate text position to center it in the cell
        const text = displayValue
        const textMetrics = this.originalCtx.measureText(text)
        const textWidth = textMetrics.width
        const textHeight = this.gridProperties.textSize

        // Center position
        const textX = destX + (cellWidth - textWidth) / 2
        const textY = destY + (cellHeight + textHeight) / 2

        // Draw text background for better visibility
        this.originalCtx.fillStyle = this.gridProperties.lineColor
        this.originalCtx.fillRect(
          textX - 2,
          textY - textHeight,
          textWidth + 4,
          textHeight + 2,
        )

        // Draw text
        this.originalCtx.fillStyle = this.gridProperties.textColor
        this.originalCtx.fillText(text, textX, textY)
      }
    }

    // Reset line dash for other drawings
    this.originalCtx.setLineDash([])

    // Draw shape if selected (real-time display)
    this.drawShapeOnGrid()
  }

  private redrawGridClassification(): void {
    if (this.backgroundImage) {
      this.drawGridClassification(
        this.backgroundImage,
        this.originalCanvas.width,
        this.originalCanvas.height,
      )
    }
  }

  private drawShapeOnGrid(): void {
    if (!this.currentShape) return

    const props = this.shapeProperties
    this.originalCtx.save()

    // Move to shape position
    this.originalCtx.translate(props.x, props.y)

    // Apply rotation (except for circle)
    if (props.type !== 'circle') {
      this.originalCtx.rotate((props.rotation * Math.PI) / 180)
    }

    // Set styles with semi-transparent opacity for overlay effect
    this.originalCtx.fillStyle = props.color
    this.originalCtx.globalAlpha = 0.5 // Semi-transparent for overlay effect
    this.originalCtx.strokeStyle = '#374151'
    this.originalCtx.lineWidth = 2
    this.originalCtx.setLineDash([]) // Solid line for shape

    // Draw shape based on type
    switch (props.type) {
      case 'square':
        const squareSize = 40 * props.size
        this.originalCtx.fillRect(
          -squareSize,
          -squareSize,
          squareSize * 2,
          squareSize * 2,
        )
        this.originalCtx.strokeRect(
          -squareSize,
          -squareSize,
          squareSize * 2,
          squareSize * 2,
        )
        break

      case 'rectangle':
        const rectWidth = props.width
        const rectHeight = props.height
        this.originalCtx.fillRect(
          -rectWidth / 2,
          -rectHeight / 2,
          rectWidth,
          rectHeight,
        )
        this.originalCtx.strokeRect(
          -rectWidth / 2,
          -rectHeight / 2,
          rectWidth,
          rectHeight,
        )
        break

      case 'circle':
        const radius = 40 * props.size
        this.originalCtx.beginPath()
        this.originalCtx.arc(0, 0, radius, 0, 2 * Math.PI)
        this.originalCtx.fill()
        this.originalCtx.stroke()
        break

      case 'triangle':
        const triangleSize = 40 * props.size
        this.originalCtx.beginPath()
        this.originalCtx.moveTo(0, -triangleSize)
        this.originalCtx.lineTo(-triangleSize * 0.866, triangleSize * 0.5)
        this.originalCtx.lineTo(triangleSize * 0.866, triangleSize * 0.5)
        this.originalCtx.closePath()
        this.originalCtx.fill()
        this.originalCtx.stroke()
        break
    }

    this.originalCtx.restore()
  }

  private selectShape(shapeType: ShapeType): void {
    // Update active button
    document
      .querySelectorAll('.shape-btn')
      .forEach(btn => btn.classList.remove('active'))
    document
      .querySelector(`[data-shape="${shapeType}"]`)
      ?.classList.add('active')

    this.currentShape = shapeType
    this.shapeProperties.type = shapeType

    // Update controls visibility
    this.updateControlsVisibility()

    // Reset shape position to center
    this.shapeProperties.x = this.editorCanvas.width / 2
    this.shapeProperties.y = this.editorCanvas.height / 2

    this.redraw()
    this.updateStatus(
      'success',
      `${
        shapeType.charAt(0).toUpperCase() + shapeType.slice(1)
      } shape selected`,
    )
  }

  private updateControlsVisibility(): void {
    const widthControl = this.elements['width-control-bottom']
    const heightControl = this.elements['height-control-bottom']
    const rotationControl = this.elements['rotation-control-bottom']
    const lockRatioCheckbox = this.elements[
      'lock-ratio-checkbox'
    ] as HTMLInputElement

    // Show rectangle-specific controls
    if (this.shapeProperties.type === 'rectangle') {
      widthControl.style.display = 'block'
      heightControl.style.display = 'block'
    } else {
      widthControl.style.display = 'none'
      heightControl.style.display = 'none'
    }

    // Sync lockRatio state with checkbox state (respect user's choice)
    if (lockRatioCheckbox) {
      this.lockRatio = lockRatioCheckbox.checked
    }

    // Hide rotation for circle
    if (this.shapeProperties.type === 'circle') {
      rotationControl.style.display = 'none'
    } else {
      rotationControl.style.display = 'block'
    }

    // Update input values
    this.updateControlValues()
  }

  private updateControlValues(): void {
    const sizeInput = this.elements['size-input-bottom'] as HTMLInputElement
    const widthInput = this.elements['width-input-bottom'] as HTMLInputElement
    const heightInput = this.elements['height-input-bottom'] as HTMLInputElement
    const rotationInput = this.elements[
      'rotation-input-bottom'
    ] as HTMLInputElement

    if (sizeInput) sizeInput.value = this.shapeProperties.size.toString()
    if (widthInput) widthInput.value = this.shapeProperties.width.toString()
    if (heightInput) heightInput.value = this.shapeProperties.height.toString()
    if (rotationInput)
      rotationInput.value = this.shapeProperties.rotation.toString()
  }

  private updateShapeProperty(
    property: keyof ShapeProperties,
    value: string | number,
  ): void {
    if (property === 'type') {
      this.shapeProperties[property] = value as ShapeType
    } else if (property === 'color') {
      this.shapeProperties[property] = value as string
    } else {
      ;(this.shapeProperties as any)[property] = value
    }
    this.redraw()
  }

  private clearShape(): void {
    this.currentShape = null
    this.shapeProperties.type = null

    // Remove active state from shape buttons
    document
      .querySelectorAll('.shape-btn')
      .forEach(btn => btn.classList.remove('active'))

    this.redraw()
    this.updateStatus('ready', 'Shape cleared')
  }

  private isPointInShape(x: number, y: number): boolean {
    if (!this.currentShape) return false

    const props = this.shapeProperties
    const dx = x - props.x
    const dy = y - props.y

    switch (props.type) {
      case 'circle':
        const radius = 40 * props.size
        return Math.sqrt(dx * dx + dy * dy) <= radius

      case 'square':
        const size = 40 * props.size
        return Math.abs(dx) <= size && Math.abs(dy) <= size

      case 'rectangle':
        return (
          Math.abs(dx) <= props.width / 2 && Math.abs(dy) <= props.height / 2
        )

      case 'triangle':
        const triangleSize = 40 * props.size
        // Simplified triangle hit detection
        return Math.abs(dx) <= triangleSize && Math.abs(dy) <= triangleSize

      default:
        return false
    }
  }

  private drawShape(): void {
    if (!this.currentShape) return

    const props = this.shapeProperties
    this.editorCtx.save()

    // Move to shape position
    this.editorCtx.translate(props.x, props.y)

    // Apply rotation (except for circle)
    if (props.type !== 'circle') {
      this.editorCtx.rotate((props.rotation * Math.PI) / 180)
    }

    // Set styles
    this.editorCtx.fillStyle = props.color
    this.editorCtx.strokeStyle = '#374151'
    this.editorCtx.lineWidth = 2

    // Draw shape based on type
    switch (props.type) {
      case 'square':
        const squareSize = 40 * props.size
        this.editorCtx.fillRect(
          -squareSize,
          -squareSize,
          squareSize * 2,
          squareSize * 2,
        )
        this.editorCtx.strokeRect(
          -squareSize,
          -squareSize,
          squareSize * 2,
          squareSize * 2,
        )
        break

      case 'rectangle':
        const rectWidth = props.width
        const rectHeight = props.height
        this.editorCtx.fillRect(
          -rectWidth / 2,
          -rectHeight / 2,
          rectWidth,
          rectHeight,
        )
        this.editorCtx.strokeRect(
          -rectWidth / 2,
          -rectHeight / 2,
          rectWidth,
          rectHeight,
        )
        break

      case 'circle':
        const radius = 40 * props.size
        this.editorCtx.beginPath()
        this.editorCtx.arc(0, 0, radius, 0, 2 * Math.PI)
        this.editorCtx.fill()
        this.editorCtx.stroke()
        break

      case 'triangle':
        const triangleSize = 40 * props.size
        this.editorCtx.beginPath()
        this.editorCtx.moveTo(0, -triangleSize)
        this.editorCtx.lineTo(-triangleSize * 0.866, triangleSize * 0.5)
        this.editorCtx.lineTo(triangleSize * 0.866, triangleSize * 0.5)
        this.editorCtx.closePath()
        this.editorCtx.fill()
        this.editorCtx.stroke()
        break
    }

    this.editorCtx.restore()

    // Draw resize handles
    this.drawResizeHandles()

    // Draw rotation handle (not for circles)
    if (props.type !== 'circle') {
      this.drawRotationHandle()
    }
  }

  private redraw(): void {
    // Clear editor canvas
    this.editorCtx.clearRect(
      0,
      0,
      this.editorCanvas.width,
      this.editorCanvas.height,
    )

    // Draw original image on editor canvas (bright for consistency) without grid
    if (this.backgroundImage) {
      this.editorCtx.save()
      this.editorCtx.globalAlpha = 1.0 // Full opacity for brightness consistency
      this.editorCtx.drawImage(
        this.backgroundImage,
        0,
        0,
        this.editorCanvas.width,
        this.editorCanvas.height,
      )
      this.editorCtx.restore()
    }

    // Draw shape if selected
    this.drawShape()

    // Update left canvas with real-time shape display
    this.redrawGridClassification()
  }

  private saveCanvas(): void {
    if (!this.backgroundImage && !this.currentShape) {
      this.updateStatus(
        'error',
        'Nothing to save. Please load an image or add a shape first.',
      )
      return
    }

    const formatInput = document.querySelector(
      'input[name="export-format"]:checked',
    ) as HTMLInputElement
    const format = (formatInput?.value as ExportFormat) || 'png'
    const mimeType = format === 'png' ? 'image/png' : 'image/jpeg'
    const quality = format === 'jpeg' ? 0.9 : 1.0

    try {
      // Create a temporary canvas for compositing
      const tempCanvas = document.createElement('canvas')
      const tempCtx = tempCanvas.getContext('2d')!

      tempCanvas.width = this.originalCanvas.width
      tempCanvas.height = this.originalCanvas.height

      // Draw original image
      if (this.backgroundImage) {
        tempCtx.drawImage(this.originalCanvas, 0, 0)
      }

      // Draw shape on top
      if (this.currentShape) {
        const props = this.shapeProperties
        tempCtx.save()
        tempCtx.translate(props.x, props.y)

        if (props.type !== 'circle') {
          tempCtx.rotate((props.rotation * Math.PI) / 180)
        }

        tempCtx.fillStyle = props.color
        tempCtx.strokeStyle = '#374151'
        tempCtx.lineWidth = 2

        switch (props.type) {
          case 'square':
            const squareSize = 40 * props.size
            tempCtx.fillRect(
              -squareSize,
              -squareSize,
              squareSize * 2,
              squareSize * 2,
            )
            tempCtx.strokeRect(
              -squareSize,
              -squareSize,
              squareSize * 2,
              squareSize * 2,
            )
            break
          case 'rectangle':
            tempCtx.fillRect(
              -props.width / 2,
              -props.height / 2,
              props.width,
              props.height,
            )
            tempCtx.strokeRect(
              -props.width / 2,
              -props.height / 2,
              props.width,
              props.height,
            )
            break
          case 'circle':
            const radius = 40 * props.size
            tempCtx.beginPath()
            tempCtx.arc(0, 0, radius, 0, 2 * Math.PI)
            tempCtx.fill()
            tempCtx.stroke()
            break
          case 'triangle':
            const triangleSize = 40 * props.size
            tempCtx.beginPath()
            tempCtx.moveTo(0, -triangleSize)
            tempCtx.lineTo(-triangleSize * 0.866, triangleSize * 0.5)
            tempCtx.lineTo(triangleSize * 0.866, triangleSize * 0.5)
            tempCtx.closePath()
            tempCtx.fill()
            tempCtx.stroke()
            break
        }
        tempCtx.restore()
      }

      tempCanvas.toBlob(
        blob => {
          if (!blob) {
            this.updateStatus('error', 'Failed to generate image')
            return
          }

          const url = URL.createObjectURL(blob)
          const a = document.createElement('a')
          a.href = url
          a.download = `graph-editor-${Date.now()}.${format}`
          document.body.appendChild(a)
          a.click()
          document.body.removeChild(a)
          URL.revokeObjectURL(url)

          this.updateStatus('success', `Image saved as ${format.toUpperCase()}`)
        },
        mimeType,
        quality,
      )
    } catch (error) {
      this.updateStatus('error', 'Failed to save image')
      console.error('Save error:', error)
    }
  }

  private updateStatus(type: StatusType, message: string): void {
    const statusText = this.elements['status-text']
    const statusIndicator = this.elements['status-indicator']

    if (statusText) {
      statusText.textContent = message
    }

    if (statusIndicator) {
      statusIndicator.className = `status-indicator ${type}`
    }

    console.log(`Status [${type}]: ${message}`)
  }

  private updateCanvasInfo(): void {
    const originalCanvasSize = this.elements['original-canvas-size']
    if (originalCanvasSize) {
      originalCanvasSize.textContent = `${this.originalCanvas.width} × ${this.originalCanvas.height}`
    }
  }

  private updateMouseCoordinates(): void {
    const mouseCoords = this.elements['mouse-coords']
    if (mouseCoords) {
      mouseCoords.textContent = `x: ${this.mousePosition.x}, y: ${this.mousePosition.y}`
    }
  }

  private async updateSpatialData() {
    let { width, height, channels } = this.imageModel.spec
    let imageTensor = tf.browser.fromPixels(this.editorCanvas, channels)
    let image = cropAndResizeImageTensor({
      imageTensor,
      width,
      height,
    })
    let x = await getImageFeatures({
      tf,
      imageModel: this.imageModel,
      image,
    })
    let array = (await x.spatialFeatures.array()) as number[][][][]
    this.spatialData = array[0]
  }

  private async updateGridData() {
    this.updateSpatialData()

    switch (this.gridDataType.type) {
      case 'feat':
        this.gridData = this.spatialData.map(row =>
          row.map(col => col[this.gridDataType.feat_num]),
        )
        break
      case 'Euclidean':
        this.gridData = this.spatialData.map(row =>
          row.map(col => {
            let sum = 0
            this.spatialData.forEach(row_i =>
              row_i.map(col_i => (sum += this.distance(col, col_i))),
            )
            return sum / 48
          }),
        )
        break
      case 'cos':
        this.gridData = this.spatialData.map(row =>
          row.map(col => {
            let sum = 0
            this.spatialData.forEach(row_i =>
              row_i.map(col_i => (sum += this.cosineSimilarity(col, col_i))),
            )
            return sum / 48
          }),
        )
        break
      default:
        break
    }

    let flattened = this.gridData.flat()

    // Calculate min and max
    const min = Math.min(...flattened)
    const max = Math.max(...flattened)

    // Calculate mean
    const mean =
      flattened.reduce((sum, value) => sum + value, 0) / flattened.length

    // Calculate standard deviation
    const variance =
      flattened.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) /
      flattened.length
    const stdDev = Math.sqrt(variance)

    this.gridStat = { min, max, mean, stdDev }
  }

  private drawResizeHandles(): void {
    if (!this.currentShape) return

    const props = this.shapeProperties
    const handleSize = 8

    let left, right, top, bottom

    switch (props.type) {
      case 'square':
        const squareSize = 40 * props.size
        left = props.x - squareSize
        right = props.x + squareSize
        top = props.y - squareSize
        bottom = props.y + squareSize
        break
      case 'rectangle':
        left = props.x - props.width / 2
        right = props.x + props.width / 2
        top = props.y - props.height / 2
        bottom = props.y + props.height / 2
        break
      case 'circle':
        const radius = 40 * props.size
        left = props.x - radius
        right = props.x + radius
        top = props.y - radius
        bottom = props.y + radius
        break
      case 'triangle':
        const triangleSize = 40 * props.size
        left = props.x - triangleSize
        right = props.x + triangleSize
        top = props.y - triangleSize
        bottom = props.y + triangleSize
        break
      default:
        return
    }

    this.editorCtx.save()
    this.editorCtx.fillStyle = '#ffffff'
    this.editorCtx.strokeStyle = '#3b82f6'
    this.editorCtx.lineWidth = 2

    // Draw corner handles
    this.drawHandle(left, top, handleSize)
    this.drawHandle(right, top, handleSize)
    this.drawHandle(left, bottom, handleSize)
    this.drawHandle(right, bottom, handleSize)

    // Draw edge handles for rectangle
    if (props.type === 'rectangle') {
      this.drawHandle(props.x, top, handleSize)
      this.drawHandle(right, props.y, handleSize)
      this.drawHandle(props.x, bottom, handleSize)
      this.drawHandle(left, props.y, handleSize)
    }

    this.editorCtx.restore()
  }

  private drawRotationHandle(): void {
    if (!this.currentShape || this.shapeProperties.type === 'circle') return

    const props = this.shapeProperties
    const handleSize = 8
    let top

    switch (props.type) {
      case 'square':
        const squareSize = 40 * props.size
        top = props.y - squareSize - 20
        break
      case 'rectangle':
        top = props.y - props.height / 2 - 20
        break
      case 'triangle':
        const triangleSize = 40 * props.size
        top = props.y - triangleSize - 20
        break
      default:
        return
    }

    this.editorCtx.save()
    this.editorCtx.fillStyle = '#10b981'
    this.editorCtx.strokeStyle = '#059669'
    this.editorCtx.lineWidth = 2

    // Draw rotation handle (circle)
    this.editorCtx.beginPath()
    this.editorCtx.arc(props.x, top, handleSize / 2, 0, 2 * Math.PI)
    this.editorCtx.fill()
    this.editorCtx.stroke()

    // Draw line connecting to shape
    this.editorCtx.strokeStyle = '#94a3b8'
    this.editorCtx.lineWidth = 1
    this.editorCtx.setLineDash([2, 2])
    this.editorCtx.beginPath()
    this.editorCtx.moveTo(props.x, top + handleSize / 2)
    this.editorCtx.lineTo(
      props.x,
      props.y -
        (props.type === 'square'
          ? 40 * props.size
          : props.type === 'rectangle'
          ? props.height / 2
          : 40 * props.size),
    )
    this.editorCtx.stroke()
    this.editorCtx.setLineDash([])

    this.editorCtx.restore()
  }

  private drawHandle(x: number, y: number, size: number): void {
    this.editorCtx.fillRect(x - size / 2, y - size / 2, size, size)
    this.editorCtx.strokeRect(x - size / 2, y - size / 2, size, size)
  }

  private distance(array1: number[], array2: number[]): number {
    if (array1.length !== array2.length) {
      throw new Error('Arrays must be of the same length')
    }
    let sum = 0
    for (let i = 0; i < array1.length; i++) {
      let diff = array1[i] - array2[i]
      sum += diff * diff
    }
    return Math.round(Math.sqrt(sum))
  }

  private cosineSimilarity(tensorA: number[], tensorB: number[]): number {
    if (tensorA.length !== tensorB.length) {
      throw new Error('Tensors must have the same length.')
    }

    const dotProduct = tensorA.reduce(
      (sum, value, index) => sum + value * tensorB[index],
      0,
    )
    const magnitudeA = Math.sqrt(
      tensorA.reduce((sum, value) => sum + value ** 2, 0),
    )
    const magnitudeB = Math.sqrt(
      tensorB.reduce((sum, value) => sum + value ** 2, 0),
    )

    if (magnitudeA === 0 || magnitudeB === 0) {
      throw new Error(
        'One or both tensors have zero magnitude, making cosine similarity undefined.',
      )
    }

    return dotProduct / (magnitudeA * magnitudeB)
  }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  try {
    new GraphEditor()
  } catch (error) {
    console.error('Failed to initialize Graph Editor:', error)
  }
})

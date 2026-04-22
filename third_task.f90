module dimensions_m
    implicit none
    integer, parameter :: nx = 1024
    integer, parameter :: ny = 1024
    integer, parameter :: TILE_DIM = 32
    integer, parameter :: BLOCK_ROWS = 8
    integer, parameter :: NUM_REPS = 100
end module dimensions_m

module kernels_m
    use cudafor
    use dimensions_m
    implicit none
contains

    attributes(global) subroutine copySharedMem(odata, idata)
        real, intent(out) :: odata(nx, ny)
        real, intent(in)  :: idata(nx, ny)
        real, shared :: tile(TILE_DIM, TILE_DIM)
        integer :: x, y, j

        x = (blockIdx%x - 1) * TILE_DIM + threadIdx%x
        y = (blockIdx%y - 1) * TILE_DIM + threadIdx%y

        do j = 0, TILE_DIM-1, BLOCK_ROWS
            tile(threadIdx%x, threadIdx%y + j) = idata(x, y + j)
        end do

        call syncthreads()

        do j = 0, TILE_DIM-1, BLOCK_ROWS
            odata(x, y + j) = tile(threadIdx%x, threadIdx%y + j)
        end do
    end subroutine copySharedMem

    attributes(global) subroutine transposeNaive(odata, idata)
        real, intent(out) :: odata(ny, nx)
        real, intent(in)  :: idata(nx, ny)
        integer :: x, y, j

        x = (blockIdx%x - 1) * TILE_DIM + threadIdx%x
        y = (blockIdx%y - 1) * TILE_DIM + threadIdx%y

        do j = 0, TILE_DIM-1, BLOCK_ROWS
            odata(x, y + j) = idata(y + j, x)
        end do
    end subroutine transposeNaive

    attributes(global) subroutine transposeCoalesced(odata, idata)
        real, intent(out) :: odata(ny, nx)
        real, intent(in)  :: idata(nx, ny)
        real, shared :: tile(TILE_DIM, TILE_DIM)
        integer :: x, y, j

        x = (blockIdx%x - 1) * TILE_DIM + threadIdx%x
        y = (blockIdx%y - 1) * TILE_DIM + threadIdx%y

        do j = 0, TILE_DIM-1, BLOCK_ROWS
            tile(threadIdx%x, threadIdx%y + j) = idata(x, y + j)
        end do

        call syncthreads()

        x = (blockIdx%y - 1) * TILE_DIM + threadIdx%x
        y = (blockIdx%x - 1) * TILE_DIM + threadIdx%y

        do j = 0, TILE_DIM-1, BLOCK_ROWS
            odata(x, y + j) = tile(threadIdx%y + j, threadIdx%x)
        end do
    end subroutine transposeCoalesced

    attributes(global) subroutine transposeNoBankConflicts(odata, idata)
        real, intent(out) :: odata(ny, nx)
        real, intent(in)  :: idata(nx, ny)
        real, shared :: tile(TILE_DIM+1, TILE_DIM)
        integer :: x, y, j

        x = (blockIdx%x - 1) * TILE_DIM + threadIdx%x
        y = (blockIdx%y - 1) * TILE_DIM + threadIdx%y

        do j = 0, TILE_DIM-1, BLOCK_ROWS
            tile(threadIdx%x, threadIdx%y + j) = idata(x, y + j)
        end do

        call syncthreads()

        x = (blockIdx%y - 1) * TILE_DIM + threadIdx%x
        y = (blockIdx%x - 1) * TILE_DIM + threadIdx%y

        do j = 0, TILE_DIM-1, BLOCK_ROWS
            odata(x, y + j) = tile(threadIdx%y + j, threadIdx%x)
        end do
    end subroutine transposeNoBankConflicts

    attributes(global) subroutine transposeDiagonal(odata, idata)
        real, intent(out) :: odata(ny, nx)
        real, intent(in)  :: idata(nx, ny)
        real, shared :: tile(TILE_DIM+1, TILE_DIM)
        integer :: x, y, j

        x = (blockIdx%x - 1) * TILE_DIM + threadIdx%x
        y = (blockIdx%y - 1) * TILE_DIM + threadIdx%y

        do j = 0, TILE_DIM-1, BLOCK_ROWS
            tile(threadIdx%x, threadIdx%y + j) = idata(x, y + j)
        end do

        call syncthreads()

        x = (blockIdx%y - 1) * TILE_DIM + threadIdx%x
        y = (blockIdx%x - 1) * TILE_DIM + threadIdx%y

        do j = 0, TILE_DIM-1, BLOCK_ROWS
            odata(x, y + j) = tile(threadIdx%y + j, threadIdx%x)
        end do
    end subroutine transposeDiagonal

end module kernels_m

program third_task
    use cudafor
    use dimensions_m
    use kernels_m
    implicit none

    type(dim3) :: grid, tBlock
    type(cudaEvent) :: startEvent, stopEvent
    type(cudaDeviceProp) :: prop
    real :: time, mem_size

    real, allocatable :: in_h(:,:), copy_h(:,:), trp_h(:,:), gold(:,:)
    real, device, allocatable :: in_d(:,:), copy_d(:,:), trp_d(:,:)

    integer :: i, j, istat

    if (mod(nx, TILE_DIM) /= 0 .or. mod(ny, TILE_DIM) /= 0) then
        print *, "ERROR: nx and ny must be multiples of TILE_DIM"
        stop
    end if
    if (mod(TILE_DIM, BLOCK_ROWS) /= 0) then
        print *, "ERROR: TILE_DIM must be a multiple of BLOCK_ROWS"
        stop
    end if

    grid = dim3(nx / TILE_DIM, ny / TILE_DIM, 1)
    tBlock = dim3(TILE_DIM, BLOCK_ROWS, 1)

    istat = cudaGetDeviceProperties(prop, 0)
    write(*, '(/, "Device Name: ", a)') trim(prop%name)
    write(*, '("Compute Capability: ", i0, ".", i0)') prop%major, prop%minor
    write(*, '(/, "Matrix size: ", i4, " x ", i4)') nx, ny
    write(*, '("Block dims: ", i3, " x ", i3)') TILE_DIM, BLOCK_ROWS
    write(*, '("Tile size: ", i3, " x ", i3)') TILE_DIM, TILE_DIM
    write(*, '("Grid: ", i4, " x ", i4, " x ", i4)') grid%x, grid%y, grid%z
    write(*, '("Thread block: ", i4, " x ", i4, " x ", i4)') tBlock%x, tBlock%y, tBlock%z

    allocate(in_h(nx, ny), copy_h(nx, ny), trp_h(ny, nx), gold(nx, ny))
    allocate(in_d(nx, ny), copy_d(nx, ny), trp_d(ny, nx))

    do j = 1, ny
        do i = 1, nx
            in_h(i, j) = real(i + (j-1) * nx)
        end do
    end do
    gold = transpose(in_h)

    in_d = in_h

    istat = cudaEventCreate(startEvent)
    istat = cudaEventCreate(stopEvent)

    mem_size = 2.0 * nx * ny * 4.0

    write(*, '(/, a25, a25)') "Routine", "Bandwidth (GB/s)"

    write(*, '(a25)', advance='no') "shared memory copy"
    copy_d = -1.0
    call copySharedMem<<<grid, tBlock>>>(copy_d, in_d)
    istat = cudaEventRecord(startEvent, 0)
    do i = 1, NUM_REPS
        call copySharedMem<<<grid, tBlock>>>(copy_d, in_d)
    end do
    istat = cudaEventRecord(stopEvent, 0)
    istat = cudaEventSynchronize(stopEvent)
    istat = cudaEventElapsedTime(time, startEvent, stopEvent)
    copy_h = copy_d
    call postprocess(copy_h, in_h, time, mem_size, NUM_REPS)

    write(*, '(a25)', advance='no') "naive transpose"
    trp_d = -1.0
    call transposeNaive<<<grid, tBlock>>>(trp_d, in_d)
    istat = cudaEventRecord(startEvent, 0)
    do i = 1, NUM_REPS
        call transposeNaive<<<grid, tBlock>>>(trp_d, in_d)
    end do
    istat = cudaEventRecord(stopEvent, 0)
    istat = cudaEventSynchronize(stopEvent)
    istat = cudaEventElapsedTime(time, startEvent, stopEvent)
    trp_h = trp_d
    call postprocess(trp_h, gold, time, mem_size, NUM_REPS)

    write(*, '(a25)', advance='no') "coalesced transpose"
    trp_d = -1.0
    call transposeCoalesced<<<grid, tBlock>>>(trp_d, in_d)
    istat = cudaEventRecord(startEvent, 0)
    do i = 1, NUM_REPS
        call transposeCoalesced<<<grid, tBlock>>>(trp_d, in_d)
    end do
    istat = cudaEventRecord(stopEvent, 0)
    istat = cudaEventSynchronize(stopEvent)
    istat = cudaEventElapsedTime(time, startEvent, stopEvent)
    trp_h = trp_d
    call postprocess(trp_h, gold, time, mem_size, NUM_REPS)

    write(*, '(a25)', advance='no') "conflict-free transpose"
    trp_d = -1.0
    call transposeNoBankConflicts<<<grid, tBlock>>>(trp_d, in_d)
    istat = cudaEventRecord(startEvent, 0)
    do i = 1, NUM_REPS
        call transposeNoBankConflicts<<<grid, tBlock>>>(trp_d, in_d)
    end do
    istat = cudaEventRecord(stopEvent, 0)
    istat = cudaEventSynchronize(stopEvent)
    istat = cudaEventElapsedTime(time, startEvent, stopEvent)
    trp_h = trp_d
    call postprocess(trp_h, gold, time, mem_size, NUM_REPS)

    write(*, '(a25)', advance='no') "diagonal transpose"
    trp_d = -1.0
    call transposeDiagonal<<<grid, tBlock>>>(trp_d, in_d)
    istat = cudaEventRecord(startEvent, 0)
    do i = 1, NUM_REPS
        call transposeDiagonal<<<grid, tBlock>>>(trp_d, in_d)
    end do
    istat = cudaEventRecord(stopEvent, 0)
    istat = cudaEventSynchronize(stopEvent)
    istat = cudaEventElapsedTime(time, startEvent, stopEvent)
    trp_h = trp_d
    call postprocess(trp_h, gold, time, mem_size, NUM_REPS)

    istat = cudaEventDestroy(startEvent)
    istat = cudaEventDestroy(stopEvent)

    deallocate(in_h, copy_h, trp_h, gold)
    deallocate(in_d, copy_d, trp_d)

contains

    subroutine postprocess(res, ref, t, mem_size, nreps)
        real, intent(in) :: res(:,:), ref(:,:), t, mem_size
        integer, intent(in) :: nreps
        real :: bandwidth
        if (all(abs(res - ref) <= 1.0e-4)) then
            bandwidth = (mem_size * 1.0e-6) / (t / real(nreps))
            write(*, '(f20.2)') bandwidth
        else
            write(*, '(a20)') "*** Failed ***"
        end if
    end subroutine postprocess

end program third_task

module matmul_kernels
    use cudafor
    implicit none
    integer, parameter :: TILE_SIZE = 32
contains
    attributes(global) subroutine matmul_naive_bad(C, A, B, m, n, k)
        real(8), device, intent(out) :: C(:,:)
        real(8), device, intent(in)  :: A(:,:), B(:,:)
        integer, value, intent(in)   :: m, n, k
        integer :: i, j, l
        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        if (i <= m .and. j <= n) then
            C(i,j) = 0.d0
            do l = 1, k
                C(i,j) = C(i,j) + A(i,l) * B(l,j)
            end do
        end if
    end subroutine matmul_naive_bad

    attributes(global) subroutine matmul_naive_good(C, A, B, m, n, k)
        real(8), device, intent(out) :: C(:,:)
        real(8), device, intent(in)  :: A(:,:), B(:,:)
        integer, value, intent(in)   :: m, n, k
        integer :: i, j, l
        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        if (i <= m .and. j <= n) then
            C(i,j) = 0.d0
            do l = 1, k
                C(i,j) = C(i,j) + A(i,l) * B(l,j)
            end do
        end if
    end subroutine matmul_naive_good

    attributes(global) subroutine matmul_tiled(C, A, B, m, n, k)
        real(8), device, intent(out) :: C(:,:)
        real(8), device, intent(in)  :: A(:,:), B(:,:)
        integer, value, intent(in)   :: m, n, k
        real(8), shared :: Asub(TILE_SIZE, TILE_SIZE)
        real(8), shared :: Bsub(TILE_SIZE, TILE_SIZE)
        integer :: i, j, p, tx, ty, bx, by, kk
        real(8) :: tmp
        tx = threadIdx%x
        ty = threadIdx%y
        bx = blockIdx%x
        by = blockIdx%y
        i = (bx-1)*TILE_SIZE + tx
        j = (by-1)*TILE_SIZE + ty
        tmp = 0.d0
        do kk = 1, k, TILE_SIZE
            if (i <= m .and. kk+ty-1 <= k) then
                Asub(tx, ty) = A(i, kk+ty-1)
            else
                Asub(tx, ty) = 0.d0
            end if
            if (j <= n .and. kk+tx-1 <= k) then
                Bsub(tx, ty) = B(kk+tx-1, j)
            else
                Bsub(tx, ty) = 0.d0
            end if
            call syncthreads()
            do p = 1, TILE_SIZE
                tmp = tmp + Asub(tx, p) * Bsub(p, ty)
            end do
            call syncthreads()
        end do
        if (i <= m .and. j <= n) then
            C(i,j) = tmp
        end if
    end subroutine matmul_tiled
end module matmul_kernels

program fourth_task
    use cudafor
    use cublas_v2
    use matmul_kernels
    implicit none
    integer, parameter :: N = 1024
    integer, parameter :: BLOCK_X = 16, BLOCK_Y = 16
    integer, parameter :: NUM_REPS = 10
    real(8), parameter :: alpha = 1.d0, beta = 0.d0
    real(8), allocatable :: A(:,:), B(:,:)
    real(8), allocatable :: C_cpu(:,:)
    real(8), allocatable :: C_bad(:,:), C_good(:,:), C_tiled(:,:), C_cublas(:,:)
    real(8), device, allocatable :: A_d(:,:), B_d(:,:)
    real(8), device, allocatable :: C_bad_d(:,:), C_good_d(:,:), C_tiled_d(:,:), C_cublas_d(:,:)
    type(dim3) :: grid_naive, tBlock_naive
    type(dim3) :: grid_tiled, tBlock_tiled
    type(cudaEvent) :: start, stop
    type(cublasHandle) :: handle
    real :: time_bad_ms, time_good_ms, time_tiled_ms, time_cublas_ms
    real(8) :: mem_size_bytes
    real(8) :: bandwidth_bad, bandwidth_good, bandwidth_tiled, bandwidth_cublas
    real(8) :: t0, t1
    integer :: i, istat

    allocate(A(N,N), B(N,N), C_cpu(N,N), stat=istat)
    if (istat /= 0) stop "Error: host allocation failed"
    allocate(C_bad(N,N), C_good(N,N), C_tiled(N,N), C_cublas(N,N), stat=istat)
    if (istat /= 0) stop "Error: host result allocation failed"
    allocate(A_d(N,N), B_d(N,N), stat=istat)
    if (istat /= 0) stop "Error: device allocation for A,B failed"
    allocate(C_bad_d(N,N), C_good_d(N,N), C_tiled_d(N,N), C_cublas_d(N,N), stat=istat)
    if (istat /= 0) stop "Error: device allocation for C matrices failed"

    call random_number(A)
    call random_number(B)

    call cpu_time(t0)
    C_cpu = matmul(A, B)
    call cpu_time(t1)
    write(*,'(/,a,f10.3,a)') "CPU matmul time: ", (t1-t0)*1000.0, " ms"

    A_d = A
    B_d = B

    tBlock_naive = dim3(BLOCK_X, BLOCK_Y, 1)
    grid_naive   = dim3(ceiling(real(N)/BLOCK_X), ceiling(real(N)/BLOCK_Y), 1)
    tBlock_tiled = dim3(TILE_SIZE, TILE_SIZE, 1)
    grid_tiled   = dim3(ceiling(real(N)/TILE_SIZE), ceiling(real(N)/TILE_SIZE), 1)

    istat = cudaEventCreate(start)
    istat = cudaEventCreate(stop)
    if (istat /= 0) stop "Error: cudaEventCreate"
    istat = cublasCreate(handle)
    if (istat /= CUBLAS_STATUS_SUCCESS) stop "Error: cublasCreate"

    mem_size_bytes = 2.0 * dble(N*N + N*N + N*N) * 8.0

    write(*,'(/,a)') "=== Matrix multiplication (N=1024, real(8)) ==="
    write(*,'(a)') "Routine                  Avg time (ms)   Bandwidth (GB/s)"
    write(*,'(a)') "---------------------------------------------------------"

    C_bad_d = 0.d0
    call matmul_naive_bad<<<grid_naive, tBlock_naive>>>(C_bad_d, A_d, B_d, N, N, N)
    istat = cudaGetLastError()
    if (istat /= 0) stop "Error: bad kernel launch (warmup)"
    istat = cudaEventRecord(start, 0)
    do i = 1, NUM_REPS
        call matmul_naive_bad<<<grid_naive, tBlock_naive>>>(C_bad_d, A_d, B_d, N, N, N)
    end do
    istat = cudaEventRecord(stop, 0)
    istat = cudaEventSynchronize(stop)
    istat = cudaEventElapsedTime(time_bad_ms, start, stop)
    if (istat /= 0) stop "Error: event timing for bad kernel"
    C_bad = C_bad_d
    bandwidth_bad = (mem_size_bytes * 1e-3) / (dble(time_bad_ms) / dble(NUM_REPS))
    write(*,'(a, f14.2, f16.2)') "naive bad access       ", &
           time_bad_ms / real(NUM_REPS), bandwidth_bad

    C_good_d = 0.d0
    call matmul_naive_good<<<grid_naive, tBlock_naive>>>(C_good_d, A_d, B_d, N, N, N)
    istat = cudaGetLastError()
    if (istat /= 0) stop "Error: good kernel launch (warmup)"
    istat = cudaEventRecord(start, 0)
    do i = 1, NUM_REPS
        call matmul_naive_good<<<grid_naive, tBlock_naive>>>(C_good_d, A_d, B_d, N, N, N)
    end do
    istat = cudaEventRecord(stop, 0)
    istat = cudaEventSynchronize(stop)
    istat = cudaEventElapsedTime(time_good_ms, start, stop)
    if (istat /= 0) stop "Error: event timing for good kernel"
    C_good = C_good_d
    bandwidth_good = (mem_size_bytes * 1e-3) / (dble(time_good_ms) / dble(NUM_REPS))
    write(*,'(a, f14.2, f16.2)') "naive good access      ", &
           time_good_ms / real(NUM_REPS), bandwidth_good

    C_tiled_d = 0.d0
    call matmul_tiled<<<grid_tiled, tBlock_tiled>>>(C_tiled_d, A_d, B_d, N, N, N)
    istat = cudaGetLastError()
    if (istat /= 0) stop "Error: tiled kernel launch (warmup)"
    istat = cudaEventRecord(start, 0)
    do i = 1, NUM_REPS
        call matmul_tiled<<<grid_tiled, tBlock_tiled>>>(C_tiled_d, A_d, B_d, N, N, N)
    end do
    istat = cudaEventRecord(stop, 0)
    istat = cudaEventSynchronize(stop)
    istat = cudaEventElapsedTime(time_tiled_ms, start, stop)
    if (istat /= 0) stop "Error: event timing for tiled kernel"
    C_tiled = C_tiled_d
    bandwidth_tiled = (mem_size_bytes * 1e-3) / (dble(time_tiled_ms) / dble(NUM_REPS))
    write(*,'(a, f14.2, f16.2)') "tiled (TILE=32)        ", &
           time_tiled_ms / real(NUM_REPS), bandwidth_tiled

    C_cublas_d = 0.d0
    istat = cublasDgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, &
                           N, N, N, alpha, A_d, N, B_d, N, beta, C_cublas_d, N)
    if (istat /= CUBLAS_STATUS_SUCCESS) stop "Error: cublas warmup"
    istat = cudaEventRecord(start, 0)
    do i = 1, NUM_REPS
        istat = cublasDgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, &
                               N, N, N, alpha, A_d, N, B_d, N, beta, C_cublas_d, N)
    end do
    istat = cudaEventRecord(stop, 0)
    istat = cudaEventSynchronize(stop)
    istat = cudaEventElapsedTime(time_cublas_ms, start, stop)
    if (istat /= 0) stop "Error: event timing for cublas"
    C_cublas = C_cublas_d
    bandwidth_cublas = (mem_size_bytes * 1e-3) / (dble(time_cublas_ms) / dble(NUM_REPS))
    write(*,'(a, f14.2, f16.2)') "cuBLAS DGEMM           ", &
           time_cublas_ms / real(NUM_REPS), bandwidth_cublas

    write(*,'(a)') "---------------------------------------------------------"
    write(*,'(/,a)') "=== Verification (max absolute difference vs CPU matmul) ==="
    write(*,'(a, es12.4)') "naive bad       : ", maxval(abs(C_bad    - C_cpu))
    write(*,'(a, es12.4)') "naive good      : ", maxval(abs(C_good   - C_cpu))
    write(*,'(a, es12.4)') "tiled           : ", maxval(abs(C_tiled  - C_cpu))
    write(*,'(a, es12.4)') "cuBLAS          : ", maxval(abs(C_cublas - C_cpu))

    istat = cublasDestroy(handle)
    istat = cudaEventDestroy(start)
    istat = cudaEventDestroy(stop)
    deallocate(A, B, C_cpu, C_bad, C_good, C_tiled, C_cublas)
    deallocate(A_d, B_d, C_bad_d, C_good_d, C_tiled_d, C_cublas_d)

end program fourth_task

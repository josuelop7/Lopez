program fifth_task
    use cudafor
    use cublas_v2
    implicit none

    integer, parameter :: M = 512
    integer, parameter :: N = 512
    integer, parameter :: K_large = 4096
    integer, parameter :: NUM_STREAMS = 4
    real(8), parameter :: alpha = 1.d0, beta = 0.d0

    real(8), allocatable :: A(:,:), B(:,:)
    real(8), allocatable :: C_serial(:,:), C_concurrent(:,:)
    real(8), device, allocatable :: A_d(:,:), B_d(:,:), C_d(:,:)

    type(cublasHandle), allocatable :: handles(:)
    integer(cuda_stream_kind), allocatable :: streams(:)
    type(cublasHandle) :: handle_ser

    real :: t0, t1, time_serial, time_concurrent
    real(8) :: flops, bandwidth, gflops_serial, gflops_conc
    real(8) :: bytes_total
    integer :: chunk_size, start_col, end_col
    integer :: i, istat, s

    write(*,'(/,a)') "=========================================="
    write(*,'(a,i0,a,i0,a,i0)') "Matrix multiplication: A(", M, ",", N, ") * B(", N, ",", K_large, ")"
    write(*,'(a,i0)') "Number of streams: ", NUM_STREAMS
    write(*,'(a)') "=========================================="

    allocate(A(M,N), B(N,K_large), stat=istat)
    if (istat /= 0) stop "Allocation failed for A, B"
    allocate(C_serial(M,K_large), C_concurrent(M,K_large), stat=istat)
    if (istat /= 0) stop "Allocation failed for C matrices"

    call random_number(A)
    call random_number(B)

    allocate(A_d(M,N), B_d(N,K_large), C_d(M,K_large), stat=istat)
    if (istat /= 0) stop "Device allocation failed"

    A_d = A
    B_d = B

    flops = 2.d0 * dble(M) * dble(N) * dble(K_large)
    bytes_total = dble(M*N + N*K_large + M*K_large) * 8.d0

    istat = cublasCreate(handle_ser)
    if (istat /= CUBLAS_STATUS_SUCCESS) stop "cublasCreate serial failed"

    call cpu_time(t0)
    istat = cublasDgemm_v2(handle_ser, CUBLAS_OP_N, CUBLAS_OP_N, &
                           M, K_large, N, alpha, A_d, M, B_d, N, beta, C_d, M)
    if (istat /= CUBLAS_STATUS_SUCCESS) stop "Serial Dgemm failed"
    istat = cudaDeviceSynchronize()
    if (istat /= 0) stop "Synchronize failed"
    call cpu_time(t1)
    time_serial = t1 - t0

    C_serial = C_d
    istat = cublasDestroy(handle_ser)
    if (istat /= CUBLAS_STATUS_SUCCESS) stop "cublasDestroy serial failed"

    allocate(handles(NUM_STREAMS), streams(NUM_STREAMS), stat=istat)
    if (istat /= 0) stop "Allocation for streams/handles failed"

    do s = 1, NUM_STREAMS
        istat = cudaStreamCreate(streams(s))
        if (istat /= 0) stop "Stream creation failed"
        istat = cublasCreate(handles(s))
        if (istat /= CUBLAS_STATUS_SUCCESS) stop "cublasCreate for stream failed"
        istat = cublasSetStream(handles(s), streams(s))
        if (istat /= CUBLAS_STATUS_SUCCESS) stop "cublasSetStream failed"
    end do

    chunk_size = K_large / NUM_STREAMS

    call cpu_time(t0)
    do s = 1, NUM_STREAMS
        start_col = (s-1) * chunk_size + 1
        if (s == NUM_STREAMS) then
            end_col = K_large
        else
            end_col = start_col + chunk_size - 1
        end if

        istat = cublasDgemm_v2(handles(s), CUBLAS_OP_N, CUBLAS_OP_N, &
                               M, end_col-start_col+1, N, &
                               alpha, A_d, M, B_d(1,start_col), N, beta, &
                               C_d(1,start_col), M)
        if (istat /= CUBLAS_STATUS_SUCCESS) then
            print *, "cublasDgemm error in stream", s
            stop
        end if
    end do

    do s = 1, NUM_STREAMS
        istat = cudaStreamSynchronize(streams(s))
        if (istat /= 0) stop "Stream sync error"
    end do
    call cpu_time(t1)
    time_concurrent = t1 - t0

    C_concurrent = C_d

    do s = 1, NUM_STREAMS
        istat = cublasDestroy(handles(s))
        if (istat /= CUBLAS_STATUS_SUCCESS) print *, "Destroy handle error", s
        istat = cudaStreamDestroy(streams(s))
        if (istat /= 0) print *, "Destroy stream error", s
    end do
    deallocate(handles, streams)

    gflops_serial = flops / (dble(time_serial)) / 1d9
    gflops_conc   = flops / (dble(time_concurrent)) / 1d9
    bandwidth = bytes_total / (dble(time_serial)) / 1d9

    write(*,'(/,a)') "────────────────────────────────────────────────────────────────────"
    write(*,'(a20,2x,a10,2x,a10,2x,a10)') "Mode", "Time(s)", "GFLOPS", "BW(GB/s)"
    write(*,'(a)') "────────────────────────────────────────────────────────────────────"
    write(*,'(a20,2x,f10.5,2x,f10.2,2x,f10.2)') "Serial", &
        time_serial, gflops_serial, bandwidth
    write(*,'(a20,2x,f10.5,2x,f10.2,2x,a10)') "Concurrent", &
        time_concurrent, gflops_conc, "    ---"
    write(*,'(a)') "────────────────────────────────────────────────────────────────────"
    write(*,'(a,f6.2)') "Speedup (concurrent/serial): ", time_serial / time_concurrent

    block
        real(8) :: max_diff
        max_diff = maxval(abs(C_serial - C_concurrent))
        write(*,'(/,a,es12.4)') "Maximum difference between serial and concurrent: ", max_diff
        if (max_diff < 1.d-10) then
            print *, "Verification PASSED"
        else
            print *, "Verification FAILED"
        end if
    end block

    deallocate(A, B, C_serial, C_concurrent, A_d, B_d, C_d)

end program fifth_task


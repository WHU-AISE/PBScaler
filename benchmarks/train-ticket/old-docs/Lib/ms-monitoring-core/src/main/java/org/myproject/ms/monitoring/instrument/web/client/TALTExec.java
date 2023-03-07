

package org.myproject.ms.monitoring.instrument.web.client;

import java.util.concurrent.Callable;
import java.util.concurrent.Future;

import org.myproject.ms.monitoring.Chainer;
import org.springframework.core.task.AsyncListenableTaskExecutor;
import org.springframework.util.concurrent.ListenableFuture;


public class TALTExec implements AsyncListenableTaskExecutor {

	private final AsyncListenableTaskExecutor delegate;
	private final Chainer tracer;

	TALTExec(AsyncListenableTaskExecutor delegate,
			Chainer tracer) {
		this.delegate = delegate;
		this.tracer = tracer;
	}

	@Override
	public ListenableFuture<?> submitListenable(Runnable task) {
		return this.delegate.submitListenable(this.tracer.wrap(task));
	}

	@Override
	public <T> ListenableFuture<T> submitListenable(Callable<T> task) {
		return this.delegate.submitListenable(this.tracer.wrap(task));
	}

	@Override
	public void execute(Runnable task, long startTimeout) {
		this.delegate.execute(this.tracer.wrap(task), startTimeout);
	}

	@Override
	public Future<?> submit(Runnable task) {
		return this.delegate.submit(this.tracer.wrap(task));
	}

	@Override
	public <T> Future<T> submit(Callable<T> task) {
		return this.delegate.submit(this.tracer.wrap(task));
	}

	@Override
	public void execute(Runnable task) {
		this.delegate.execute(this.tracer.wrap(task));
	}

}
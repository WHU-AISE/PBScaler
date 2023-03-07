

package org.myproject.ms.monitoring.instrument.async;

import java.util.concurrent.Callable;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

import org.myproject.ms.monitoring.ItemNamer;
import org.myproject.ms.monitoring.Chainer;
import org.myproject.ms.monitoring.ChainKeys;


public class TSEServ extends TEServ implements ScheduledExecutorService {

	public TSEServ(ScheduledExecutorService delegate,
			Chainer tracer, ChainKeys traceKeys, ItemNamer spanNamer) {
		super(delegate, tracer, traceKeys, spanNamer);
	}

	private ScheduledExecutorService getScheduledExecutorService() {
		return (ScheduledExecutorService) this.delegate;
	}

	@Override
	public ScheduledFuture<?> schedule(Runnable command, long delay, TimeUnit unit) {
		Runnable r = new SCTRun(this.tracer, this.traceKeys, this.spanNamer, command);
		return getScheduledExecutorService().schedule(r, delay, unit);
	}

	@Override
	public <V> ScheduledFuture<V> schedule(Callable<V> callable, long delay, TimeUnit unit) {
		Callable<V> c = new SCTCall<>(this.tracer, this.traceKeys, this.spanNamer,  callable);
		return getScheduledExecutorService().schedule(c, delay, unit);
	}

	@Override
	public ScheduledFuture<?> scheduleAtFixedRate(Runnable command, long initialDelay, long period, TimeUnit unit) {
		Runnable r = new SCTRun(this.tracer, this.traceKeys, this.spanNamer,  command);
		return getScheduledExecutorService().scheduleAtFixedRate(r, initialDelay, period, unit);
	}

	@Override
	public ScheduledFuture<?> scheduleWithFixedDelay(Runnable command, long initialDelay, long delay, TimeUnit unit) {
		Runnable r = new SCTRun(this.tracer, this.traceKeys, this.spanNamer,  command);
		return getScheduledExecutorService().scheduleWithFixedDelay(r, initialDelay, delay, unit);
	}

}

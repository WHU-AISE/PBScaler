

package org.myproject.ms.monitoring.instrument.web.client;

import java.io.IOException;
import java.net.URI;

import org.myproject.ms.monitoring.instrument.web.HSInject;
import org.myproject.ms.monitoring.Chainer;
import org.myproject.ms.monitoring.instrument.web.HTKInject;
import org.springframework.core.task.AsyncListenableTaskExecutor;
import org.springframework.http.HttpMethod;
import org.springframework.http.client.AsyncClientHttpRequest;
import org.springframework.http.client.AsyncClientHttpRequestFactory;
import org.springframework.http.client.ClientHttpRequest;
import org.springframework.http.client.ClientHttpRequestFactory;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.scheduling.concurrent.ThreadPoolTaskScheduler;


public class TACHRFW extends ATHRInter
		implements ClientHttpRequestFactory, AsyncClientHttpRequestFactory {

	final AsyncClientHttpRequestFactory asyncDelegate;
	final ClientHttpRequestFactory syncDelegate;

	
	public TACHRFW(Chainer tracer,
			HSInject spanInjector,
			AsyncClientHttpRequestFactory asyncDelegate,
			HTKInject httpTraceKeysInjector) {
		super(tracer, spanInjector, httpTraceKeysInjector);
		this.asyncDelegate = asyncDelegate;
		this.syncDelegate = asyncDelegate instanceof ClientHttpRequestFactory ?
				(ClientHttpRequestFactory) asyncDelegate : defaultClientHttpRequestFactory();
	}

	
	public TACHRFW(Chainer tracer,
			HSInject spanInjector, HTKInject httpTraceKeysInjector) {
		super(tracer, spanInjector, httpTraceKeysInjector);
		SimpleClientHttpRequestFactory simpleClientHttpRequestFactory = defaultClientHttpRequestFactory();
		this.asyncDelegate = simpleClientHttpRequestFactory;
		this.syncDelegate = simpleClientHttpRequestFactory;
	}

	public TACHRFW(Chainer tracer,
			HSInject spanInjector,
			AsyncClientHttpRequestFactory asyncDelegate,
			ClientHttpRequestFactory syncDelegate,
			HTKInject httpTraceKeysInjector) {
		super(tracer, spanInjector, httpTraceKeysInjector);
		this.asyncDelegate = asyncDelegate;
		this.syncDelegate = syncDelegate;
	}

	private SimpleClientHttpRequestFactory defaultClientHttpRequestFactory() {
		SimpleClientHttpRequestFactory simpleClientHttpRequestFactory = new SimpleClientHttpRequestFactory();
		simpleClientHttpRequestFactory.setTaskExecutor(asyncListenableTaskExecutor(this.tracer));
		return simpleClientHttpRequestFactory;
	}

	private AsyncListenableTaskExecutor asyncListenableTaskExecutor(Chainer tracer) {
		ThreadPoolTaskScheduler threadPoolTaskScheduler = new ThreadPoolTaskScheduler();
		threadPoolTaskScheduler.initialize();
		return new TALTExec(threadPoolTaskScheduler, tracer);
	}

	@Override
	public AsyncClientHttpRequest createAsyncRequest(URI uri, HttpMethod httpMethod)
			throws IOException {
		AsyncClientHttpRequest request = this.asyncDelegate
				.createAsyncRequest(uri, httpMethod);
		addRequestTags(request);
		publishStartEvent(request);
		return request;
	}

	@Override
	public ClientHttpRequest createRequest(URI uri, HttpMethod httpMethod)
			throws IOException {
		ClientHttpRequest request = this.syncDelegate.createRequest(uri, httpMethod);
		addRequestTags(request);
		publishStartEvent(request);
		return request;
	}
}
